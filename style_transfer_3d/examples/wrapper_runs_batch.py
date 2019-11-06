from configargparse import ArgParser
from time import time
from os.path import join as pjoin
#import subprocess
from subprocess import Popen, PIPE
import pandas as pd
def main(args):
    #### fix content file names:
    cfiles = [pjoin(args.data_dir,'meshes', cf) for cf in args.content_fns] 
    #### fix style file names:
    sfiles = [pjoin(args.data_dir,'styles', sf) for sf in args.style_fns] 
    
    output = dict()

    for cf in cfiles:
        cn = cf.split("/")[-1]
        for sf in sfiles:
            
            sn = sf.split("/")[-1]

            t = time()
            row = cn+"_"+sn
            #print(subprocess.call('python ./examples/run.py -im %s.obj -is %s.jpg -o %s.gif -lc %d -ltv %d' % (cf,sf,pjoin(args.out_dir,row),args.lc,args.ltv) ,shell=True))
            #content, style, tv = subprocess.STDOUT 

            p = Popen(["python ./examples/run.py -im %s.obj -is %s.jpg -o %s.gif -lc %d -ltv %d" % (cf,sf,pjoin(args.out_dir,row),args.lc,args.ltv)], stdout=PIPE,shell=True)
            #content, style, tv, loss = 0,0,0,0
            #print(" *** Return Code : %d" %(p.returncode))

            #cleaning string input 
            content, style, tv, loss  = tuple([float(x.strip()) for x in p.stdout.read().replace("variable","").replace("(","").replace(")","").split(",")])

            output[row] =  {'content_loss':content,
                            'style_loss':style,
                            'tv_loss':tv,
                            'overall_loss': loss,
                            'time':  int(time()-t) }
            
    print(output)
    #output
    pd.DataFrame.from_dict(output,orient="index").to_csv(pjoin(args.out_dir,"style_output.csv"))

if __name__ == "__main__":
    p = ArgParser()
    p.add('-c', '--config_file', default='config.yml', is_config_file=True, help='Path to config file.')
    p.add('--content_fns', type=str, nargs='+', help='List of variables.')
    p.add('--style_fns', type=str, nargs='+', help='List of variables.')
    p.add('--data_dir', type=str, help='Directory containing examples')
    p.add('--out_dir', type=str, help='Directory where print output')
    p.add('--lc', type=int, help='lc')
    p.add('--ltv', type=int, help='ltv')

    args = p.parse_args()
    main(args)


