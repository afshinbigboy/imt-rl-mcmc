import scipy as sp
import numpy as np
from scipy import stats
import networkx as nx
from networkx.algorithms.traversal.depth_first_search import dfs_tree
from matplotlib import pyplot as plt
import matplotlib as mpl
import random
import matplotlib.image as mpimg
import graphviz
import imageio, json
from IPython.display import Image
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import matplotlib.patches as mpatches



font = {
    'weight' : 'normal',
    'size'   : 13,
}
mpl.rc('font', **font)



def plot_mat(M, row='', col='', title='', save_name=None):
    rows, cols = M.shape[:2]
    plt.imshow(M, cmap='GnBu', interpolation="nearest")
    plt.yticks(range(M.shape[0]), ['%s %d'%(row,i) for i in range(rows)])
    plt.xticks(range(M.shape[1]), ['%s %d'%(col,i) for i in range(cols)])
    plt.xticks(rotation=60)
    plt.xlabel('{}-{} Matrix'.format(row.capitalize(), col.capitalize()))
    plt.title(title)
    if save_name:
        plt.savefig(save_name)
    plt.imshow()
    plt.close()
    

class Tree(object):
    def __init__(self, T, raw_T, E, E_l, CP, lossed_genes, **params):
        self.__T = T
        self.__raw_T = raw_T
        self.__E = E
        self.__E_l = E_l
        self.__CP = CP
        self.__N = E.shape[0]
        self.__M = E.shape[1]
        params['N'] = self.__N
        params['M'] = self.__M
        self.params = params
        self.lossed_genes = lossed_genes
        self.psi = params['psi']
        self.vartheta = params['vartheta']
        self.varrho = params['varrho']

        self.__plot_scale = 30./max(self.__M, self.__N)
        
        self.__set_params(params)
        self.generate_data(**params)

        
    def generate_data(self, **params):
        self.__new_param = self.__params
        for k,v in params.items():
            self.__new_param[k]=v
        
        if not json.dumps(self.__params) == json.dumps(self.__new_param):
            print('Prev params:')
            print('\t'.join(json.dumps(self.__params, indent=True).splitlines()))
            self.__set_params(self.__new_param)
            print('New params:')
            print('\t'.join(json.dumps(params, indent=True).splitlines()))

        ## ========================================================
        ## ~~~~~~~~~~~~~~~~~~~~~~~~ E to D ~~~~~~~~~~~~~~~~~~~~~~~~
        ## ========================================================
        D = self.__E.copy()
        nz_idxs = np.nonzero(self.__E)
        z_idxs  = np.nonzero(self.__E-1)
        z_rnds  = np.random.rand(len( z_idxs[0]))
        nz_rnds = np.random.rand(len(nz_idxs[0]))
        z_rnds  = [1 if i < self.__alpha  else 0 for i in  z_rnds]
        nz_rnds = [0 if i < self.__beta   else 1 for i in nz_rnds]
        D[nz_idxs] = nz_rnds
        D[ z_idxs] =  z_rnds
        self.__D = D
        
        ## ========================================================
        ## ~~~~~~~~~~~~~~~~~~~~~~ E_l to D_l ~~~~~~~~~~~~~~~~~~~~~~
        ## ========================================================
#         D_l = self.__E_l.copy()
        D_l = np.where(self.__E_l < 1, 0, 1)
        nz_idxs = np.nonzero(self.__E_l)
        z_idxs  = np.nonzero(self.__E_l-1)
        z_rnds  = np.random.rand(len( z_idxs[0]))
        nz_rnds = np.random.rand(len(nz_idxs[0]))
        z_rnds  = [1 if i < self.__alpha  else 0 for i in  z_rnds]
        nz_rnds = [0 if i < self.__beta   else 1 for i in nz_rnds]
        D_l[nz_idxs] = nz_rnds
        D_l[ z_idxs] =  z_rnds
        self.__D_l = D_l
        
        ## ========================================================
        ## ~~~~~~~~~~~~~~~~ add missing data Dm ~~~~~~~~~~~~~~~~~~~
        ## ========================================================
        Dm = self.__D.copy()
        idxs = np.nonzero(self.__D+1)
        rnds = np.random.rand(self.__N, self.__M)
        for n in range(self.__N):
            for m in range(self.__M):
                if rnds[n, m] < self.__MR:
                    Dm[n, m] = 3
        self.__Dm = Dm
        
        ## ========================================================
        ## ~~~~~~~~~~~~~~~~ add missing data Dlm ~~~~~~~~~~~~~~~~~~
        ## ========================================================
        Dlm = self.__D_l.copy()
        idxs = np.nonzero(self.__D_l+1)
        rnds = np.random.rand(self.__N, self.__M)
        for n in range(self.__N):
            for m in range(self.__M):
                if rnds[n, m] < self.__MR:
                    Dlm[n, m] = 3
        self.__Dlm = Dlm
        
        
    def __set_params(self, params):
        self.__alpha = params['alpha']
        self.__beta  = params['beta']
        self.__MR    = params['MR'] # missing rate
        
        self.__params = params
        self.__str_params  ='_'.join(['{}={}'.format(k,v) for k,v in params.items()])
        self.__latex_params='\ '.join(['{}={}'.format(k if len(k)<3 else '\%s'%k,v) for k,v in params.items()])
        
        llp = len(self.__latex_params)//2-4
        self.__latex_params = f"{self.__latex_params[:llp]}$\n${self.__latex_params[llp:]}"
        
        
    def save_data(self, save_dir):
        if not save_dir[-1]=='/':
            save_dir += '/'
        p = 'Parameters: {}\n'.format(self.__str_params)
        np.savetxt('{}E.csv'.format(save_dir), self.__E, fmt='%.0f', delimiter=',', header=p)
        np.savetxt('{}D.csv'.format(save_dir), self.__D, fmt='%.0f', delimiter=',', header=p)
        np.savetxt('{}DmE.csv'.format(save_dir), self.__D-self.__E, fmt='%.0f', delimiter=',', header=p)
        np.savetxt('{}Dm.csv'.format(save_dir), self.__Dm, fmt='%.0f', delimiter=',', header=p)
    
                       
    def get_E(self,):
        return self.__E
    
    def get_D(self,):
        return self.__D
                       
    def get_Dm(self,):
        return self.__Dm
    
    def get_T(self,):
        return self.__T
    
    def get_raw_T(self,):
        return self.__raw_T
     
                       
    def get_params(self,):
        return self.__params
        
                       
    def get_alpha(self,):
        return self.__alpha
    
    
    def get_beta(self,):
        return self.__beta
    
    
    def get_mcmc_tree_data(self,):
        return (self.__E, self.__D, self.__Dm, self.__CP, self.__raw_T)
        return (self.__E_l, self.__D_l, self.__Dlm, self.__CP, self.T)
    
    
    def save_tree(self, save_path):
        file_path = '{}Tree_{}.gpickle'.format(save_path, self.__str_params) if save_path[-1] == '/' else save_path
        nx.write_gpickle(self.__T, file_path)
    
    
    def plot_tree_mut(self, save_path):
        mut_T = self.__T.copy()
#         mut_T.remove_nodes_from([i for i,n in enumerate(self.__T.nodes()) if 'cell' in str(n)])
        pdot = nx.drawing.nx_pydot.to_pydot(mut_T)
        file_path = '{}treeM_{}.png'.format(save_path, self.__str_params) if save_path[-1] == '/' else save_path
        pdot.write_png(file_path)
        
        
    def plot_tree_full(self, save_path, title=None):
        pdot = nx.drawing.nx_pydot.to_pydot(self.__T)
        for i, node in enumerate(pdot.get_nodes()):
            node_name = str(node)[:-1]
            if 'cell' in node_name:
                node.set_label('s%s'%node_name.split()[-1][:-1])
                node.set_shape('egg')
                node.set_fillcolor('#db8625')
                node.set_color('red')
        file_path = f'{save_path}treeF_{self.__str_params}.png' if save_path[-1] == '/' else save_path
        pdot.write_png(file_path)
        if title: print(title)
        

    def plot_E(self, save_path=None, nofig=False, figsize=None):
        if not nofig:
            plt.figure(figsize=figsize if figsize else (self.__M*self.__plot_scale,self.__N*self.__plot_scale))
        plt.imshow(self.__E, cmap='GnBu', interpolation="nearest")
        plt.yticks(range(self.__E.shape[0]), ['cell %d'%i for i in range(self.__N)])
        plt.xticks(range(self.__E.shape[1]), [ 'mut %d'%i for i in range(self.__M)])
        plt.xticks(rotation=60)
        plt.xlabel('Genes-Cells Matrix E (Error-less)')
        plt.title(r'P: ${}$'.format(self.__latex_params))
        if save_path is not None:
            file_path = '{}E_{}.png'.format(save_path, self.__str_params) if save_path[-1] == '/' else save_path
            plt.savefig(file_path)
            plt.close()
            return imageio.imread(file_path)
        if not nofig:
            plt.show()
            plt.close()
            
    
    def plot_E_l(self, save_path=None, nofig=False, figsize=None):
        if not nofig:
            plt.figure(figsize=figsize if figsize else (self.__M*self.__plot_scale,self.__N*self.__plot_scale))
        ## first you need to define your color map and value name as a dict
        t = 1 ## alpha value
        cmap = {0:[1,1,0.95,t], 1:[0.25,0.25,0.5,t], 0.5:[0.9,0.25,0.5,t]}
        labels = {0:'0', 1:'1', 0.5:'lost'}
        arrayShow = np.array([[cmap[i] for i in j] for j in 0.5*(self.__E_l + self.__E)])    
        ## create patches as legend
        patches =[mpatches.Patch(color=cmap[i],label=labels[i]) for i in cmap]
        plt.imshow(arrayShow, interpolation="nearest")
        plt.legend(handles=patches, loc=2, borderaxespad=-6)
        plt.yticks(range(self.__E.shape[0]), ['cell %d'%i for i in range(self.__N)])
        plt.xticks(range(self.__E.shape[1]), [ 'mut %d'%i for i in range(self.__M)])
        plt.xticks(rotation=60)
        plt.xlabel('$E with lossed mutation$')
        plt.title(r'P: ${}$'.format(
            "\ ".join([
                f"N={self.__N}", f"M={self.__M}", 
                f"\zeta={self.params['zeta']}", f"\gamma={self.params['gamma']}"
                f"\psi={self.psi}", f"\vartheta={self.vartheta}", f"\varrho={self.varrho}", 
            ])
        ))
        if save_path is not None:
            file_path = '{}El_{}.png'.format(save_path, self.__str_params) if save_path[-1] == '/' else save_path
            plt.savefig(file_path)
            plt.close()
            return imageio.imread(file_path)
        if not nofig:
            plt.show()
            plt.close()
    
    
    def plot_D(self, save_path=None, nofig=False, figsize=None):
        if not nofig:
            plt.figure(figsize=figsize if figsize else (self.__M*self.__plot_scale,self.__N*self.__plot_scale))
        plt.imshow(self.__D, cmap='GnBu', interpolation="nearest")
        plt.yticks(range(self.__D.shape[0]), ['cell %d'%i for i in range(self.__N)])
        plt.xticks(range(self.__D.shape[1]), [ 'mut %d'%i for i in range(self.__M)])
        plt.xticks(rotation=60)
        plt.xlabel('Noisy Genes-Cells Matrix D (input Data)')
        plt.title(r'P: ${}$'.format(self.__latex_params))
        if save_path is not None:
            file_path = '{}D_{}.png'.format(save_path, self.__str_params) if save_path[-1] == '/' else save_path
            plt.savefig(file_path)
            plt.close()
            return imageio.imread(file_path)
        if not nofig:
            plt.show()
            plt.close()


    def plot_D_l(self, save_path=None, nofig=False, figsize=None):
        if not nofig:
            plt.figure(figsize=figsize if figsize else (self.__M*self.__plot_scale,self.__N*self.__plot_scale))
        plt.imshow(self.__D_l, cmap='GnBu', interpolation="nearest")
        plt.yticks(range(self.__D_l.shape[0]), ['cell %d'%i for i in range(self.__N)])
        plt.xticks(range(self.__D_l.shape[1]), [ 'mut %d'%i for i in range(self.__M)])
        plt.xticks(rotation=60)
        plt.xlabel('Noisy Genes-Cells Matrix D_l (input Data)')
        plt.title(r'P: ${}$'.format(self.__latex_params))
        if save_path is not None:
            file_path = '{}Dl_{}.png'.format(save_path, self.__str_params) if save_path[-1] == '/' else save_path
            plt.savefig(file_path)
            plt.close()
            return imageio.imread(file_path)
        if not nofig:
            plt.show()
            plt.close()

        
    def plot_DmE(self, save_path=None, nofig=False, figsize=None):
        if not nofig:
            plt.figure(figsize=figsize if figsize else (self.__M*self.__plot_scale,self.__N*self.__plot_scale))
        ## first you need to define your color map and value name as a dict
        t = 1 ## alpha value
        cmap = {0:[1,1,0.95,t], 1:[0.5,0.5,0.8,t], -1:[0.8,0.5,0.5,t]}
        labels = {0:'true', 1:'false positive', -1:'false negetive'}
        arrayShow = np.array([[cmap[i] for i in j] for j in self.__D-self.__E])    
        ## create patches as legend
        patches =[mpatches.Patch(color=cmap[i],label=labels[i]) for i in cmap]
        plt.imshow(arrayShow, interpolation="nearest")
        plt.legend(handles=patches, loc=2, borderaxespad=-6)
        plt.yticks(range(self.__E.shape[0]), ['cell %d'%i for i in range(self.__N)])
        plt.xticks(range(self.__E.shape[1]), [ 'mut %d'%i for i in range(self.__M)])
        plt.xticks(rotation=60)
        plt.xlabel('D-E')
        plt.title(r'P: ${}$'.format(self.__latex_params))
        if save_path is not None:
            file_path = '{}DmE_{}.png'.format(save_path, self.__str_params) if save_path[-1] == '/' else save_path
            plt.savefig(file_path)
            plt.close()
            return imageio.imread(file_path)
        if not nofig:
            plt.show()
            plt.close()
            
    def plot_DlmEl(self, save_path=None, nofig=False, figsize=None):
        if not nofig:
            plt.figure(figsize=figsize if figsize else (self.__M*self.__plot_scale,self.__N*self.__plot_scale))
        ## first you need to define your color map and value name as a dict
        t = 1 ## alpha value
        cmap = {0:[1,1,0.95,t], 1:[0.5,0.5,0.8,t], -1:[0.8,0.5,0.5,t]}
        labels = {0:'true', 1:'false positive', -1:'false negetive'}
        arrayShow = np.array([[cmap[i] for i in j] for j in self.__D_l-self.__E_l])    
        ## create patches as legend
        patches =[mpatches.Patch(color=cmap[i],label=labels[i]) for i in cmap]
        plt.imshow(arrayShow, interpolation="nearest")
        plt.legend(handles=patches, loc=2, borderaxespad=-6)
        plt.yticks(range(self.__E_l.shape[0]), ['cell %d'%i for i in range(self.__N)])
        plt.xticks(range(self.__E_l.shape[1]), [ 'mut %d'%i for i in range(self.__M)])
        plt.xticks(rotation=60)
        plt.xlabel('Dl-El')
        plt.title(r'P: ${}$'.format(self.__latex_params))
        if save_path is not None:
            file_path = '{}DlmEl_{}.png'.format(save_path, self.__str_params) if save_path[-1] == '/' else save_path
            plt.savefig(file_path)
            plt.close()
            return imageio.imread(file_path)
        if not nofig:
            plt.show()
            plt.close()
    
    
    def plot_Dm(self, save_path=None, nofig=False, figsize=None):
        if not nofig:
            plt.figure(figsize=figsize if figsize else (self.__M*self.__plot_scale,self.__N*self.__plot_scale))
        ## first you need to define your color map and value name as a dict
        t = 1 ## alpha value
        cmap = {0:[1,1,0.95,t], 1:[0.2,0.2,0.4,t], 3:[0.8,0.5,0.5,t]}
        labels = {0:'0', 1:'1', 3:'missed'}
        arrayShow = np.array([[cmap[i] for i in j] for j in self.__Dm])    
        ## create patches as legend
        patches =[mpatches.Patch(color=cmap[i],label=labels[i]) for i in cmap]

        plt.imshow(arrayShow, interpolation="nearest")
        plt.legend(handles=patches, loc=2, borderaxespad=-6)
        plt.yticks(range(self.__D.shape[0]), ['cell %d'%i for i in range(self.__N)])
        plt.xticks(range(self.__D.shape[1]), [ 'mut %d'%i for i in range(self.__M)])
        plt.xticks(rotation=60)
        plt.xlabel('Noisy Genes-Cells Matrix with Missed Data ($D_m$)')
        plt.title(r'P: ${}$'.format(self.__latex_params))
        if save_path is not None:
            file_path = '{}Dm_{}.png'.format(save_path, self.__str_params) if save_path[-1] == '/' else save_path
            plt.savefig(file_path)
            plt.close()
            return imageio.imread(file_path)
        if not nofig:
            plt.show()
            plt.close()
            
    def plot_Dlm(self, save_path=None, nofig=False, figsize=None):
        if not nofig:
            plt.figure(figsize=figsize if figsize else (self.__M*self.__plot_scale,self.__N*self.__plot_scale))
        ## first you need to define your color map and value name as a dict
        t = 1 ## alpha value
        cmap = {0:[1,1,0.95,t], 1:[0.2,0.2,0.4,t], 3:[0.8,0.5,0.5,t]}
        labels = {0:'0', 1:'1', 3:'missed'}
        arrayShow = np.array([[cmap[i] for i in j] for j in self.__Dlm])    
        ## create patches as legend
        patches =[mpatches.Patch(color=cmap[i],label=labels[i]) for i in cmap]

        plt.imshow(arrayShow, interpolation="nearest")
        plt.legend(handles=patches, loc=2, borderaxespad=-6)
        plt.yticks(range(self.__D_l.shape[0]), ['cell %d'%i for i in range(self.__N)])
        plt.xticks(range(self.__D_l.shape[1]), [ 'mut %d'%i for i in range(self.__M)])
        plt.xticks(rotation=60)
        plt.xlabel('Noisy Genes-Cells Matrix with Missed Data ($Dl_m$)')
        plt.title(r'P: ${}$'.format(self.__latex_params))
        if save_path is not None:
            file_path = '{}Dlm_{}.png'.format(save_path, self.__str_params) if save_path[-1] == '/' else save_path
            plt.savefig(file_path)
            plt.close()
            return imageio.imread(file_path)
        if not nofig:
            plt.show()
            plt.close()
    
    def plot_all_mat(self, figsize=None):
        figsize = figsize if figsize else (self.__M*self.__plot_scale,self.__N*self.__plot_scale)
        plt.figure(figsize=figsize)
        plt.subplot(2, 2, 1)
        plt.title('E')
        self.plot_E(figsize=np.asarray(figsize)/2, nofig=True)
        plt.subplot(2, 2, 2)
        plt.title('D')
        self.plot_D(figsize=np.asarray(figsize)/2, nofig=True)
        plt.subplot(2, 2, 3)
        plt.title('D-E')
        self.plot_DmE(figsize=np.asarray(figsize)/2, nofig=True)
        plt.subplot(2, 2, 4)
        plt.title('Dm')
        self.plot_Dm(figsize=np.asarray(figsize)/2, nofig=True)
        plt.show()
        plt.close()
        
        plt.figure(figsize=figsize)
        plt.subplot(2, 2, 1)
        plt.title('E_l')
        self.plot_E_l(figsize=np.asarray(figsize)/2, nofig=True)
        plt.subplot(2, 2, 2)
        plt.title('D_l')
        self.plot_D_l(figsize=np.asarray(figsize)/2, nofig=True)
        plt.subplot(2, 2, 3)
        plt.title('D_l-E_l')
        self.plot_DlmEl(figsize=np.asarray(figsize)/2, nofig=True)
        plt.subplot(2, 2, 4)
        plt.title('Dlm')
        self.plot_Dlm(figsize=np.asarray(figsize)/2, nofig=True)
        plt.show()
        plt.close()