import numpy as np
from scipy import stats
import networkx as nx
from networkx.algorithms.traversal.depth_first_search import dfs_tree
import random
from .ds import Tree




class TreeGenerator():
    '''
    Inputs: (M, N, ZETA, Gamma, alpha, beta, MR, save_images, save_trees, save_mats)
        --------------------------------------
        M       :   num of genes (mutations)
        --------------------------------------
        N       :   num of samples (cells)
        --------------------------------------
        ZETA    :   homogeness of tree
        --------------------------------------
        Gamma   :   merge genes
        --------------------------------------
        alpha   :   ~ P(D=1|E=0)
        --------------------------------------
        beta    :   ~ P(D=0|E=1)
        --------------------------------------
        MR      :   missing ratio
        --------------------------------------
        psi     :   Loss probablity of a mutation in T
        --------------------------------------
        vartheta:   A parameter related to distanse
        --------------------------------------
        varrho  :   Probablity of rejecting lossable genes respect to decreased copy number
        --------------------------------------
        

    Outputs: (E, D, Dm, T)
        ------------------------------------------------
        E       :   Mutation-cell matrix without errors
        ------------------------------------------------
        D       :   Mutation-cell matrix with errors
        ------------------------------------------------
        Dm      :   <D> with missed data
        ------------------------------------------------
        T       :   The generated tree
        ------------------------------------------------
    '''

    def __init__(self,
        M, N, 
        ZETA=1, Gamma=0.15, alpha=0.001,beta=0.08, MR=0.05,
        psi = 0.7, vartheta = 0.2, varrho = 0.2,
        cnp_length=5, cnp_init_min=2, cnp_init_max=5, cnp_end_max=8, cnp_end_min=0,
        save_dir=None):

        self.M=M
        self.N=max(M, N)
        self.ZETA=ZETA
        self.Gamma=Gamma
        self.alpha=alpha
        self.beta=beta
        self.MR=MR
        self.psi=psi
        self.vartheta=vartheta
        self.varrho=varrho
        self.cnp_length=cnp_length
        self.cnp_init_min=cnp_init_min
        self.cnp_init_max=cnp_init_max
        self.cnp_end_max=cnp_end_max
        self.cnp_end_min=cnp_end_min
        self.save_dir=save_dir


    def generate(self,):
        ## ========================================================
        ## ~~~~~~~~~~~~~~~~ generate a random tree ~~~~~~~~~~~~~~~~
        ## ========================================================
        self.Tree = dict()
        self.cnt = 2
        xk = np.arange(self.M+1)
        name_k = [str(i) for i in xk]
        wk = np.ones(self.M+1, dtype=np.float128)
        while True:
            xk, wk, name_k, u, v = self.do_next(xk, wk, name_k)
            self.cnt+=1
            if len(xk) < 2:
                break
        T = nx.DiGraph(self.Tree)
        T_leaves = [x for x in T.nodes() if T.out_degree(x)==0 and T.in_degree(x)==1]
        T.remove_nodes_from(T_leaves)
        t = np.arange(self.M)
        np.random.shuffle(t)
        t = dict((i, j) for i,j in zip(T.nodes(), t))
        T = nx.relabel_nodes(T, t)
        raw_T = T.copy()
        root = [n for n,d in raw_T.in_degree() if d==0][0]
        
        ## ========================================================
        ## ~~~~~~~~~~~~~~~ merge some of mutations ~~~~~~~~~~~~~~~~
        ## ========================================================
        A = int(np.floor(self.Gamma*self.M))
        if A:
            for i in range(A):
                while True:
                    p, c = random.sample(T.edges(),1)[0]
                    if p != root: break
                for child in T.successors(c):
                    T.add_edge(p,child)        
                T.remove_node(c)
                T = nx.relabel_nodes(T, {p: '{} . {}'.format(p,c)})

        ## ========================================================
        ## ~~~~~~~~~~~~~~~~~ add cells to the tree ~~~~~~~~~~~~~~~~
        ## ========================================================
        Mutaions_T = T.copy()
        mutaion_nodes = Mutaions_T.nodes()
        cells = np.array(['cell %d'%i for i in range(self.N)])
        np.random.shuffle(cells)
        for n in mutaion_nodes:
            T.add_edge(n, cells[0])
            cells = cells[1:]
        for cell in cells:
            node = random.sample(mutaion_nodes, 1)[0]
            T.add_edge(node, cell)
        
        ## ========================================================
        ## ~~~~~~~~~~~~~~~~~~~~~~ Tree to E ~~~~~~~~~~~~~~~~~~~~~~~
        ## ========================================================
        E = np.zeros([self.N, self.M])        
        E[int(root), :] = 1        
        for n in range(self.N):
            try:
                path = list(nx.all_simple_paths(T, root, 'cell %d'%n))[0]
            except:
                print('root:', root)
                pdot = nx.drawing.nx_pydot.to_pydot(T)
                pdot.write_png('problem_tree.png')
                exit()
            for g in path[:-1]:
                try:
                    E[n, int(g)] = 1
                except:
                    gs = g.split(' . ')
                    for g in gs:
                        E[n, int(g)] = 1

        ## ========================================================
        ## ~~~~~~~~~~~~ perform acceptable losses (CP) ~~~~~~~~~~~~
        ## ========================================================
#         '''
#         0. Prpare list of links named `all_L`
#         1. Choose a random link L_i:(M_u->M_v) from `all_L` if their contain at least one sample individualy.
#             In fact we choose two samples (S_u, S_v).
#         2. Choose a set of mutations in ancestors of `M_v` named M_xs.
#         3.  [a] Add an attribute to selected link L_i:(loss M_x).
#             [b] For each cell_i that contains M_v, triger M_x to 0.
#             [c] Write "L(S_u,_Sv): M_xs" in the CP_gt.txt file.
#             [d] Write "L(S_u,_Sv): M_xs,<some random Ms in [M in path (v->root)]-[M_xs]>" in the CP.txt file.
#         4. Remove L_i from `all_L`
#         5. If it is not enough go to step 1.
#         6. Repeat Y times above loop to achieve additional L(i,j). 
#             But in this case append them just to the CP.txt file.
#         '''

        gene_cnp_idx = np.random.randint(0, self.cnp_length, self.M)
        initial_cnp = np.random.randint(self.cnp_init_min, self.cnp_init_max, self.cnp_length)

        root = [n for n,d in T.in_degree() if d==0][0]
        cnp_dict = {
            'nodes': {
                str(root): initial_cnp
            },
            'cells': {}
        }
        lossable_genes_all = {}
        for node in nx.traversal.bfs_tree(T, root):
            if node == root: continue
            node_str = str(node)
            for parent in T.predecessors(node):
                break
            parent_cnp = cnp_dict['nodes'][str(parent)]
            if 'cell' in node_str:
                cnp_dict['cells'][node_str] = parent_cnp
            else:
                new_cnp, lossable_genes = self.__gen_new_cnp(parent_cnp, gene_cnp_idx)
                cnp_dict['nodes'][node_str] = new_cnp
                if lossable_genes and np.random.rand() > self.varrho:
                    valid_lossable_genes = []
                    dist = 0
                    for parent in nx.algorithms.shortest_path(T.to_undirected(), node, root):
                        if parent == node: continue
                        dist += 1
                        if '.' in str(parent):
                            parents = [int(p) for p in parent.split(' . ')]
                            for parent in parents:
                                if parent in lossable_genes:
                                    valid_lossable_genes.append((parent, dist))
                        else:
                            if parent in lossable_genes:
                                valid_lossable_genes.append((parent, dist))

                    loss_point = node_str
                    for valid_lossable_gen, dist in valid_lossable_genes:
                        if str(valid_lossable_gen) in lossable_genes_all.keys():
                            lossable_genes_all[str(valid_lossable_gen)].append({'loss_point': loss_point, 'distance': dist})
                        else:
                            lossable_genes_all[str(valid_lossable_gen)] = [{'loss_point': loss_point, 'distance': dist}]

        self.lossed_genes = []
        for lossable_gen, values in lossable_genes_all.items():
            if np.random.rand() > self.psi:
                distances = [v['distance'] for v in values]
                chooseable_points = [v['loss_point'] for v in values]
                choosed_idx = np.random.randint(0, len(distances))
                choosed_distance = distances[choosed_idx]
                choosed_loss_point = chooseable_points[choosed_idx]

                self.lossed_genes.append({
                    "gene": str(lossable_gen),
                    "loss_point": choosed_loss_point
                })

        self.CP = np.zeros([self.N, self.cnp_length])
        for cell, cnp in cnp_dict['cells'].items():
            cell_idx = int(cell.split(' ')[-1])
            for j, cn in enumerate(cnp):
                self.CP[cell_idx, j] = cn

        ## ========================================================
        ## ~~~~~~~~~~~~~~ E_l: E with lossed genes ~~~~~~~~~~~~~~~~
        ## ========================================================
        E_l = E.copy()
        
        for lossed_gene in self.lossed_genes:
            gene = lossed_gene["gene"]
            loss_point = lossed_gene["loss_point"]
            
            try:
                sT = dfs_tree(T, int(loss_point))
            except:
                sT = dfs_tree(T, loss_point)
            cells = [n for n in sT.nodes() if sT.out_degree(n)==0 and sT.in_degree(n)==1]
            for cell in cells:
                cell_idx = int(cell.split(" ")[-1])
                E_l[cell_idx, int(gene)] = 0
#                 E[cell_idx, int(gene)] = 0.5
        
        ## ========================================================
        ## ~~~~~~~~~~~~~~~~ return generated data ~~~~~~~~~~~~~~~~~
        ## ========================================================
        tree_obj = Tree(
            T               = T.copy(), 
            raw_T           = raw_T.copy(), 
            E               = E, 
            E_l             = E_l, 
            CP              = self.CP,
            zeta            = self.ZETA,
            gamma           = self.Gamma,
            lossed_genes    = self.lossed_genes,
            psi             = self.psi,
            vartheta        = self.vartheta,
            varrho          = self.varrho,
            alpha           = self.alpha,
            beta            = self.beta,
            MR              = self.MR,
        )
        return tree_obj
        # return (E.astype(int), D.astype(int), Dm.astype(int), raw_T, tree_obj)


    def do_next(self, xk, wk, name_k):
        u, v = self.__weighted_drand(xk, wk, size=2)
        idx_u = np.where(xk==u)[0]
        idx_v = np.where(xk==v)[0]
        w_u = wk[idx_u]
        w_v = wk[idx_v]
        w_uv = (w_u+w_v)/(self.ZETA**0.25)
        nu = name_k[int(idx_u)]
        nv = name_k[int(idx_v)]
        nuv = '{}.{}'.format(nu, nv)
        self.Tree[nuv] = [nu, nv]
        xk = np.delete(xk, [idx_u, idx_v])
        name_k = np.delete(name_k, [idx_u, idx_v])
        wk = np.delete(wk, [idx_u, idx_v])
        xk = np.append(xk, self.M+self.cnt)
        name_k = np.append(name_k, nuv)
        wk = np.append(wk, w_uv)
        return (xk, wk, name_k, u, v)
    
    @staticmethod
    def __rand_pmf(xk, pk, size=1):
        custm = stats.rv_discrete(name='custm', values=(xk, pk))
        cnt = 0
        while True:
            rs = custm.rvs(size = size)
            if len(set(rs)) == len(rs):
                break
            cnt+=1
        return rs

    def __weighted_drand(self, xk, wk, size=1):
        pk = wk/np.sum(wk, dtype=np.float128)
        return self.__rand_pmf(xk, pk, size)
    
    def __gen_new_cnp(self, cnp, gene_cnp_idx):
            new_cnp = []
            lossable_genes = []
            for i, cn in enumerate(cnp):
                rnd = np.random.rand()
                if rnd < 0.1:
                    new_cnp.append(max(self.cnp_end_min, cn-1))
                    lossable_genes = np.where(gene_cnp_idx == i)[0].tolist()
                elif rnd > 0.85:
                    new_cnp.append(min(self.cnp_end_max, cn+1))
                else:
                    new_cnp.append(cn)
            return new_cnp, lossable_genes