import distance
from apted import APTED, Config
from apted.helpers import Tree
from lxml import html
from collections import deque
from parallel import parallel_process
import numpy as np
import subprocess
import re
import os
import sys
from html import escape

class TableTree(Tree):
    def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)

    def bracket(self):
        """Show tree using brackets notation"""
        if self.tag == 'td':
            result = '"tag": %s, "colspan": %d, "rowspan": %d, "text": %s' % \
                     (self.tag, self.colspan, self.rowspan, self.content)
        else:
            result = '"tag": %s' % self.tag
        for child in self.children:
            result += child.bracket()
        return "{{{}}}".format(result)

class CustomConfig(Config):
    @staticmethod
    def maximum(*sequences):
        """Get maximum possible value
        """
        return max(map(len, sequences))

    def normalized_distance(self, *sequences):
        """Get distance from 0 to 1
        """
        return float(distance.levenshtein(*sequences)) / self.maximum(*sequences)

    def rename(self, node1, node2):
        """Compares attributes of trees"""
        if (node1.tag != node2.tag) or (node1.colspan != node2.colspan) or (node1.rowspan != node2.rowspan):
            return 1.
        if node1.tag == 'td':
            if node1.content or node2.content:
                return self.normalized_distance(node1.content, node2.content)
        return 0.

def tokenize(node):
    ''' Tokenizes table cells
    '''
    global __tokens__
    __tokens__.append('<%s>' % node.tag)
    if node.text is not None:
        __tokens__ += list(node.text)
    for n in node.getchildren():
        tokenize(n)
    if node.tag != 'unk':
        __tokens__.append('</%s>' % node.tag)
    if node.tag != 'td' and node.tail is not None:
            __tokens__ += list(node.tail)

def format_html(tags, rev_word_map_tags, cells=None, rev_word_map_cells=None):
    ''' Formats html code from raw model output
    '''
    HTML = [rev_word_map_tags[ind] for ind in tags[1:-1]]
    if cells is not None:
        to_insert = [i for i, tag in enumerate(HTML) if tag in ('<td>', '>')]
        for i, cell in zip(to_insert[::-1], cells[::-1]):
            if cell is not None:
                cell = [rev_word_map_cells[ind] for ind in cell[1:-1]]
                cell = ''.join([escape(token) if len(token) == 1 else token for token in cell])
                HTML.insert(i + 1, cell)

    HTML = '''<html>
    <head>
    <meta charset="UTF-8">
    <style>
    table, th, td {
      border: 1px solid black;
      font-size: 10px;
    }
    </style>
    </head>
    <body>
    <table frame="hsides" rules="groups" width="100%%">
    %s
    </table>
    </body>
    </html>''' % ''.join(HTML)
    return HTML

def tree_convert_html(node, convert_cell=False, parent=None):
    ''' Converts HTML tree to the format required by apted
    '''
    global __tokens__
    if node.tag == 'td':
        if convert_cell:
            __tokens__ = []
            tokenize(node)
            cell = __tokens__[1:-1].copy()
        else:
            cell = []
        new_node = TableTree(node.tag,
                             int(node.attrib.get('colspan', '1')),
                             int(node.attrib.get('rowspan', '1')),
                             cell, *deque())
    else:
        new_node = TableTree(node.tag, None, None, None, *deque())
    if parent is not None:
        parent.children.append(new_node)
    if node.tag != 'td':
        for n in node.getchildren():
            tree_convert_html(n, convert_cell, new_node)
    if parent is None:
        return new_node

def similarity_eval_html(pred, true, structure_only=False):
    ''' Computes TEDS score between the prediction and the ground truth of a
        given samples
    '''
    if pred.xpath('body/table') and true.xpath('body/table'):
        pred = pred.xpath('body/table')[0]
        true = true.xpath('body/table')[0]
        n_nodes_pred = len(pred.xpath(".//*"))
        n_nodes_true = len(true.xpath(".//*"))
        tree_pred = tree_convert_html(pred, convert_cell=not structure_only)
        tree_true = tree_convert_html(true, convert_cell=not structure_only)
        n_nodes = max(n_nodes_pred, n_nodes_true)
        distance = APTED(tree_pred, tree_true, CustomConfig()).compute_edit_distance()
        return 1.0 - (float(distance) / n_nodes)
    else:
        return 0.0

def TEDS_wraper(prediction, ground_truth, filename=None):
    if prediction:
        return similarity_eval_html(
            html.fromstring(prediction),
            html.fromstring(ground_truth)
        )
    else:
        return 0.

def TEDS(gt, pred, n_jobs=8):
    ''' Computes TEDS scores for an evaluation set
    '''
    assert n_jobs > 0 and isinstance(n_jobs, int), 'n_jobs must be positive integer'
    inputs = [{'filename': filename, 'prediction': pred.get(filename, ''), 'ground_truth': attributes['html']} for filename, attributes in gt.items()]
    scores = parallel_process(inputs, TEDS_wraper, use_kwargs=True, n_jobs=n_jobs, front_num=1)
    scores = {i['filename']: score for i, score in zip(inputs, scores)}
    return scores

def html2xml(html_code, out_path):
    if not html_code:
        return
    root = html.fromstring(html_code)
    if root.xpath('body/table'):
        table = root.xpath('body/table')[0]
        cells = []
        multi_row_cells = []
        row_pt = 0
        for row in table.iter('tr'):
            row_skip = np.inf
            col_pt = 0
            for cell in row.getchildren():
                # Skip cells expanded from previous rows
                multi_row_cells = sorted(multi_row_cells, key=lambda x: x['start-col'])
                for c in multi_row_cells:
                    if 'end-col' in c:
                        if c['start-row'] <= row_pt <= c['end-row'] and c['start-col'] <= col_pt <= c['end-col']:
                            col_pt += c['end-col'] - c['start-col'] + 1
                    else:
                        if c['start-row'] <= row_pt <= c['end-row'] and c['start-col'] == col_pt:
                            col_pt += 1
                # Generate new cell
                new_cell = {'start-row': row_pt,
                            'start-col': col_pt,
                            'content': html.tostring(cell, method='text', encoding='utf-8').decode('utf-8')}
                # Handle multi-row/col cells
                if int(cell.attrib.get('colspan', '1')) > 1:
                    new_cell['end-col'] = col_pt + int(cell.attrib['colspan']) - 1
                if int(cell.attrib.get('rowspan', '1')) > 1:
                    new_cell['end-row'] = row_pt + int(cell.attrib['rowspan']) - 1
                    multi_row_cells.append(new_cell)
                if new_cell['content']:
                    cells.append(new_cell)
                row_skip = min(row_skip, int(cell.attrib.get('rowspan', '1')))
                col_pt += int(cell.attrib.get('colspan', '1'))
            row_pt += row_skip if not np.isinf(row_skip) else 1
            multi_row_cells = [cell for cell in multi_row_cells if row_pt <= cell['end-row']]
        with open(out_path, 'w') as fp:
            fp.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            fp.write('<document>\n')
            fp.write('  <table id=\'0\'>\n')
            fp.write('    <region id=\'0\' page=\'0\' col-increment=\'0\' row-increment=\'0\'>\n')
            for i, cell in enumerate(cells):
                attributes = ' '.join(['%s=\'%d\'' % (key, value) for key, value in cell.items() if key != 'content'])
                fp.write('      <cell id=\'%d\' %s>\n' % (i, attributes))
                fp.write('        <content>%s</content>\n' % escape(cell['content']))
                fp.write('      </cell>\n')
            fp.write('    </region>\n')
            fp.write('  </table>\n')
            fp.write('</document>')

def relation_metric(pred, gt, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.6, 0.95, 8)
    precisions = []
    recalls = []
    f1scores = []
    for threshold in thresholds:
        try:
            result = subprocess.check_output(['java', '-jar', 'dataset-tools-fat-lib.jar', '-str', gt, pred, '-threshold%f' % threshold])
            result = result.split(b'\n')[-2].decode('utf-8')
            try:
                precision = float(re.search(r'Precision[^=]*= ([0-9.]*)', result).group(1))
            except ValueError:
                print(ValueError, file=sys.stderr)
                precision = 0.0
            try:
                recall = float(re.search(r'Recall[^=]*= ([0-9.]*)', result).group(1))
            except ValueError:
                print(ValueError, file=sys.stderr)
                recall = 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0. else 0.
            precisions.append(precision)
            recalls.append(recall)
            f1scores.append(f1)
        except Exception as e:
            print(os.path.basename(pred), file=sys.stderr)
            print(e, file=sys.stderr)
            precisions.append(0.)
            recalls.append(0.)
            f1scores.append(0.)
    return np.mean(precisions), np.mean(recalls), np.mean(f1scores)


if __name__ == '__main__':
    from paramiko import SSHClient

    html_pred = '/Users/peterzhong/Downloads/table2html/Tag+Cell/PMC5059900_003_02.html'
    with open(html_pred, 'r') as fp:
        pred = html.parse(fp).getroot()
    filename = os.path.basename(html_pred).split('.')[0]

    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect('dccxl003.pok.ibm.com', username='peterz')
    sftp_client = ssh.open_sftp()
    with sftp_client.open('/dccstor/ddig/peter/Medline_paper_annotator/data/table_norm/htmls/%s.html' % (filename)) as remote_file:
        true = html.parse(remote_file).getroot()
        true_table = html.Element("table")
        for n in true.xpath('body')[0].getchildren():
            true_table.append(n)
        true.xpath('body')[0].append(true_table)
    print(similarity_eval_html(pred, true))
