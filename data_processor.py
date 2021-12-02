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

After conversation reconstruction, avg # messages per conversation is 73.
Excluding convos with less than 10 messages (spam, grubhut, etc.)
the avg becomes 170. Max 2135, Min 11, Total 19093. Pretty decent length!

12/2/2021
I noticed that some emojis are showing up in the text, these will (maybe) be 
removed in the preprocessing layer for simplicity's sake
"""
from xml.etree import ElementTree

# function to determine how to format the training data
def text_formatter(single_message, type):
    if type == '1':
        return 'You:\n' + single_message + '\n'
    elif type == '2':
        return 'Brian:\n' + single_message + '\n'
    else:
        raise('The SMS message does not have a type of 1 or 2.')

# conversation reconstruction from XML
def get_dataset(xml_path):

    # load up the texts
    tree = ElementTree.parse(xml_path)
    root = tree.getroot()

    convos = {} # each convo is indexed by phone number of correspondent
    for child in root:

        # only process SMS texts
        if child.tag != 'sms':
            continue

        # find this text's phone number
        phone_num = child.get('address')
        
        # get the training data style representation
        text_to_add = text_formatter(child.get('body'), child.get('type'))

        if phone_num in convos:
            # append text to the convo
            convos[phone_num] += text_to_add
        else:
            # create new convo entry in the dict
            convos[phone_num] = text_to_add
        
    # smush the convos together to form one big dataset of continuous text
    dataset = ''.join(convos.values())

    return dataset

if __name__ == '__main__':
    dataset = get_dataset('text_messages.xml')

print('debug')