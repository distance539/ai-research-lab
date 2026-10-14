"""Independent orchestration reusing frozen095/096 definitions, no upstream copy."""
WORDS=['negative','positive']
def question(row):
    return row['sentence']+'\nQuestion: Is this sentence positive or negative?\nAnswer:'
def block(row):
    return question(row)+' '+WORDS[row['label']]+'\n\n'
def specification(name,row,train,plan):
    if name in plan['conditions']:
        return ''.join(block(train[i]) for i in plan['conditions'][name])+question(row),WORDS,[0,1]
    if name=='short_text':
        return 'Review: '+row['sentence']+'\nSentiment:',WORDS,[0,1]
    assert name.startswith('listed_')
    mapping=[1,0] if name.endswith('reverse') else [0,1]
    ctx=row['sentence']+'\nQuestion: Is this sentence positive or negative?\n'
    ctx+='\n'.join(f'{a}. {WORDS[y]}' for a,y in zip('AB',mapping))+'\nAnswer:'
    return ctx,list('AB') if 'letter' in name else [WORDS[y] for y in mapping],mapping
