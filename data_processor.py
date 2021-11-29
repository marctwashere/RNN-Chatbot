"""
11/29/2021
Script reads that the texts are distributed as such: {'sms': 19482, 'mms': 864}
Probably for my purposes, I will discard the MMS and only use SMS
as training data. Basically don't want to deal with pics/video/groupchat.

Just from context, the attrib 'type' determines if phone received/sent.
Type counts from XML analysis (SMS and MMS): {'2': 9734, '1': 9748, None: 864}
Type counts for SMS only: {'2': 9734, '1': 9748}

From context clues:
'type' == 1 means phone RECEIVED the message
'type' == 2 means phone SENT the message
"""
from xml.etree import ElementTree

# load up the texts
tree = ElementTree.parse('text_messages.xml')
root = tree.getroot()

# get some statistics
tag_names = {}
for child in root:
    if child.tag != 'sms':
        continue

    tag = child.get('type')
    if tag in tag_names:
        tag_names[tag] += 1
    else:
        tag_names[tag] = 1
        print('Found new type of tag! {}'.format(tag))
    
    if tag is None:
        print(child.get('body'))
print(tag_names)

print('debug')