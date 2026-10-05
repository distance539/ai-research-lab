"""Fixed instruction variants; source text and class mapping remain byte-identical."""
from common import message
from mixture import news_message

def messages(c,name,text):
    if name=='json_original':return message(c,'json',text)
    if name=='plain_original':return message(c,'plain',text)
    if name=='news_original':return news_message(c,text)
    if name=='json_paraphrase':
        user='Decide whether the sentiment expressed in this movie review is positive or negative.\n'+c['contracts']['json']+'\nReview: '+text
    elif name=='json_reordered':
        user=c['contracts']['json']+'\n'+c['task']+'\nReview: '+text
    elif name=='news_paraphrase':
        user='Assign this news article to one category. A = World; B = Sports; C = Business; D = Science and Technology.\n'+c['news_contract']+'\nArticle: '+text
    else:raise ValueError(name)
    return [{'role':'system','content':c['system']},{'role':'user','content':user}]

def contract(name):return 'letter' if name.startswith('news') else ('plain' if name.startswith('plain') else 'json')
