'''
Generates html to qualitatively validate predictions, in the following format
    Prediction       Ground Truth
-------------------+-------------------
|       Table      |       Table      |
-------------------+-------------------
            PDF table image
         ---------------------
         |     Table image   |
         ---------------------
'''
from lxml import html, etree

def generate_comparison(html_pred, html_gt, image_file, output_html, similarity=None, structure_only=False):
    # Add more html style settings
    html_pred.xpath('head/style')[0].text += \
        '''figure {
          display: block;
          margin-top: 1em;
          margin-bottom: 1em;
          margin-left: 40px;
          margin-right: 40px;
          text-align: center;
        }
        .floatedTable {
            float:left;
        }
        .inlineTable {
            display: inline-block;
        }
        '''
    # Add table caption
    title = html.Element("caption")
    if similarity is None:
        title.text = 'Prediction'
    else:
        title.text = 'Prediction (similarity: %.1f%%)' % (100 * similarity)

    # Alter table setting
    for table in html_pred.xpath('body/table'):
        table.insert(0, title)
        table.attrib['width'] = '49%'
        table.attrib['class'] = 'inlineTable'

    if structure_only:
        # Replace cell content with dummy text
        for node in html_gt.xpath('body/table/thead/tr/td'):
            etree.strip_tags(node, '*')
            node.tag = 'th'
            node.text = 'Dummy header'
        # Replace cell content with dummy text
        for node in html_gt.xpath('body/table/tbody/tr/td'):
            etree.strip_tags(node, '*')
            node.text = 'Dummy body cell'

    # Insert ground truth table
    gt_table = html_gt.xpath('body/table')[0]
    gt_table.attrib['class'] = 'inlineTable'
    gt_table.attrib['frame'] = "hsides"
    gt_table.attrib['rules'] = "groups"
    gt_table.attrib['width'] = "49%"
    title = html.Element("caption")
    title.text = 'Ground Truth'
    gt_table.insert(0, title)
    html_pred.xpath('body')[0].append(gt_table)

    # Insert the original table image
    figure = html.Element('figure')
    img = html.Element("img", src=image_file)
    img.attrib['class'] = 'center'

    caption = html.Element("figcaption")
    caption.text = 'PDF table image'
    figure.append(caption)
    figure.append(img)
    html_pred.xpath('body')[0].append(figure)

    # Save new html
    for n in html_pred.iter('bold'):
        n.tag = 'b'
    for n in html_pred.iter('italic'):
        n.tag = 'i'
    str = html.tostring(html_pred, pretty_print=True, method="html")
    with open(output_html, 'wb') as fp:
        fp.write(str)


if __name__ == '__main__':
    from paramiko import SSHClient
    from metric import similarity_eval_html
    from glob import glob
    import os

    structure_only = False
    html_dir = '/Users/peterzhong/Downloads/table2html/%s/*.html' % ('Tag' if structure_only else 'Tag+Cell')
    html_files = glob(html_dir)
    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect('dccxl003.pok.ibm.com', username='peterz')
    sftp_client = ssh.open_sftp()

    for html_file in html_files:
        with open(html_file, 'r') as fp:
            root = html.parse(fp).getroot()
        filename = os.path.basename(html_file).split('.')[0]
        try:
            # Load ground truth from CCC
            with sftp_client.open('/dccstor/ddig/peter/Medline_paper_annotator/data/table_norm/htmls/%s.html' % (filename)) as remote_file:
                content = remote_file.read().decode('utf-8')
                gt_root = html.document_fromstring(content)
            true_table = html.Element("table")
            for n in gt_root.xpath('body')[0].getchildren():
                true_table.append(n)
            gt_root.xpath('body')[0].append(true_table)
            similarity = similarity_eval_html(root, gt_root, structure_only)
            generate_comparison(root, gt_root, "%s.png" % filename, html_file, similarity, structure_only)
        except Exception:
            # Continue if file does not exist
            raise
