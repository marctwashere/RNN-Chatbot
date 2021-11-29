from xml.etree import ElementTree

# load up the texts
tree = ElementTree.parse('text_messages.xml')
root = tree.getroot()

# get some statistics
tag_names = {}
for child in root:
    tag = child.tag
    if tag in tag_names:
        tag_names[tag] += 1
    else:
        tag_names[tag] = 1
        print('Found new type of tag! {}'.format(tag))
print(tag_names)

print('debug')