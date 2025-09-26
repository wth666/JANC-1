def interface_x(fv):
    fm2 = fv[:,0:-5]
    fm1 = fv[:,1:-4]
    f = fv[:,2:-3]
    fp1 = fv[:,3:-2]
    fp2 = fv[:,4:-1]
    fp3 = fv[:,5:]
    f_halfp = (3*fm2-25*fm1+150*f+150*fp1-25*fp2+3*fp3)/256
    return f_halfp