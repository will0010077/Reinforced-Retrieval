
s=list(map(int,list(input())))

def getmexnumber(s):
    a=sorted(s)
    mex=0
    for ele in a:
        if ele>=mex:
            mex+=1
    return mex
    
    

print(getmexnumber(s))
