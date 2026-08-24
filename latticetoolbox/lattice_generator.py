import sys
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pickle
from matplotlib.lines import Line2D
import os
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

from latticetoolbox.lattice_sets import n_latt_triangular_very_large

#my_path = os.path.join(os.path.abspath(os.path.dirname(os.path.realpath(__file__))), '../../')

# triangular lattices with eccentricity 1- 1.25
lARGE_IDS = [1000010,
             1100011,
             1200012,
             1300013,
             1400014,
             1500015,
             1600016,
             1700017,
             1800018,
             1900019,
             2000020]

s3 = np.array([[1, 0], [0, -1]], dtype=complex)

def where_vec_array(arr, vec):
    return np.where(np.isclose(arr, vec).all(axis=1))[0]

class Triangular:
    def __init__(self, n, id, order="paper"):
        """

        :param n: Number of sites
        :param id: Lattice ID
        """

        # number of sites
        self.n = n

        # lattice constant
        self.a = 1.

        # lattice vectors
        self.a1 = np.array([self.a, 0]).transpose()
        self.a2 = np.array([self.a/2., np.sqrt(3)*self.a/2]).transpose()
        self.base_vectors = [np.array([0., 0.]).transpose()]

        # Lattice ID as in LC
        # fill utlf matrix from lattice id
        if id not in lARGE_IDS:
            self.id = str(id)
            utlf_d = int(self.id[-2::])
            utlf_b = int(self.id[-4::][:2])
            utlf_a = int(self.id[:len(self.id) - 4])
            # UTLF Matrix, Simulation Torus as in LC
            self.utlf = np.zeros(shape=(2, 2), dtype=float)
        else:
            self.id = str(id)
            utlf_d = int(self.id[-2::])
            utlf_b = 0.
            utlf_a = int(self.id[:2])

        assert utlf_b < utlf_d  # see lattice catalogue pdf
        self.utlf = np.array([[utlf_a, utlf_b], [0, utlf_d]], dtype=float)

        self.simulation_torus = np.zeros(shape=(2, 2), dtype=float)
        # get simulation torus from utlf matrix
        self.simulation_torus = np.column_stack([utlf_a * self.a1 + utlf_b * self.a2, utlf_d * self.a2])

        # (x,y) coordinates for each site with site 0 at (0,0) -> therefore with -1 initialized because
        # we check at some point if coordinates already in array
        self.coordinates = np.full((self.n, 2), -1.)

        # node ids which will be tuple (row_index, column_index)
        self.node_ids = np.zeros(shape=(self.n, 2), dtype=int)

        # adjacency matrix of graph
        self.adjacencymatrix = np.full((self.n, self.n), -1)

        # order of indices of lattice sites
        self.order = order
        # order list, will be a mapping index->value thus default is index=value
        self.new_order = np.arange(self.n)
        """
        Note:
            for a site i with coordinates self.coordinates[i] the corresponding Majorana label 
            is np.where(self.new_order = i)
        """

        # list of sites in the horizontal direction
        self.horizontal_strings = []

        # in the "up-right" direction
        self.diagonal_strings_ur = []

        # in the "up-left" direction
        self.diagonal_strings_ul = []

        # t_1 and t_2 vectors for lattices where the vectors are not the components of the utlf matrix but "geometrically equivalent"
        # n: {Majorana_Number: {Lattice ID: [[t1[0] t1[0]],[[t1[0], t2[1]]}}
        self.sim_tor_dict = {4: {10104 : [[1, 1],[2, -2]]},
                             5: {10105: [[1, 1],[2, -3]]},
                             6: {10106: [[1,1], [3, -3]],
                                 10206: [[1,2],[2, -2]]},
                             7: {10207: [[1, 2], [3, -1]],
                                 10107: [[1, 1], [3, -4]]},
                             8: {10208: [[1, 2], [3, -2]],
                                 10108: [[1,1],[4, -4]],
                                 10308: [[1,3],[2, -2]]},
                             9: {10109: [[1,1],[4, -5]],
                                 10209: [[1,2],[4, -1]]},
                             10: {10110: [[1,1],[5, -5]],
                                  10210: [[1,2],[4, -2]],
                                  10410: [[2, -2],[3,2]]},
                             11: {10111: [[1,1],[5, -6]],
                                  10211: [[1,2],[4, -3]]},
                             12: {20206: [[2, 2],[2, -4]],
                                  10412: [[1,4],[3, 0]],
                                  10512: [[2, -2],[3,3]],
                                  10212: [[1,2],[5, -2]],
                                  10112: [[1,1],[5,-7]]},
                             13: {10313: [[1,3],[4, -1]],
                                  10113: [[1, 1],[5, -8]],
                                  10213: [[1,2],[5, -3]]},
                             14: {10114: [[1,1],[5, -9]],
                                  10314: [[1, 3], [4, -2]],
                                  10214: [[1,2],[5, -4]],
                                  10614: [[2, -2], [3, 4]]},
                             15: {10315: [[1,3],[4, -3]],
                                  10415: [[1,4],[4,1]],
                                  10115: [[1,1],[5, -10]],
                                  10215: [[1,2],[5, -5]],
                                  10515: [[1,5],[3, 0]]},
                             16: {20208: [[2,2],[4, -4]],
                                  10616: [[2, -4],[3, 2]],
                                  10716: [[2, -2], [3, 5]],
                                  10116: [[1, 1],[5, -11]],
                                  10316: [[1,3],[5, -1]],
                                  10216: [[1, 2],[5, -6]]},
                             17: {10117: [[1,1],[5, -12]],
                                  10317: [[1,3],[5, -2]],
                                  10217: [[1,2],[5, -7]]},
                             18: {10318: [[1,3],[5, -3]],
                                  10418: [[1,4],[4, -2]],
                                  10218: [[1,2],[5, -8]],
                                  10518: [[3, -3], [4, 2]],
                                  10818: [[2,2],[5, 4]]},
                             19: {10719: [[2, -5],[3, 2]]},
                             20: {10420: [[1, 4], [5, 0]],
                                  10320: [[1,3], [5, -5]],
                                  10520: [[1, 5],[4,0]],
                                  10820: [[2, -4],[3, 4]]},
                             21: {10421: [[1, 4],[5, -1]],
                                  10721: [[1, 7], [3, 0]],
                                  10621: [[3, -3],[4, 3]]},
                             22: {10422: [[1,4],[5, -2]]},
                             23: {10423: [[1, 4], [5, -3]]},
                             24: {10424: [[1,4],[5, -4]],
                                  20212: [[2,2],[6, -6]]},
                             25: {50005: [[5,0],[0, 5]]},
                             27: {30309: [[3,3],[3, -6]]},
                             28: {20414: [[2, 4],[6, -2]]},
                             30: {11130: [[3,3],[5, -5]]},
                             31: {10531: [[1, 5], [5, -6]]},
                             32: {20416: [[2,4],[6, -4]],
                                  40008: [[4, 0],[0,8]]},
                             36: {60006: [[6,0],[0,6]]},
                             39: {11639: [[2, -7],[5, 2]]},
                             42: {10742: [[1,7],[5, -7]]},
                             45: {11945: [[2, -7], [5,5]]},
                             48: {40412: [[4,4],[4, -8]]},
                             49: {70007: [[7,0], [0, 7]]},
                             #196: {1400014: [[14,0],[14,0]]},
                             }

        # either get t1, t2 vectors from simulation torus or from the dicitonary
        try:
            mat = self.sim_tor_dict[self.n][int(self.id)]

            t1_comp = np.array(mat[0]).transpose()
            t2_comp = np.array(mat[1]).transpose()
            # print("dict used instead of utlf matrix for t_1 and t_2 vectors")

            l0 = [int(el) for el in [0., t1_comp[0], t2_comp[0], t1_comp[0] + t2_comp[0]]]
            max_n1 = np.amax(l0)
            min_n1 = np.amin(l0)

            l1 = [int(el) for el in [0., t1_comp[1], t2_comp[1], t1_comp[1] + t2_comp[1]]]
            max_n2 = np.amax(l1)
            min_n2 = np.amin(l1)

            self.t1 = t1_comp[0] * self.a1 + t1_comp[1] * self.a2
            self.t2 = t2_comp[0] * self.a1 + t2_comp[1] * self.a2

        except KeyError:
            self.t1 = self.simulation_torus[:, 0]
            self.t2 = self.simulation_torus[:, 1]

            max_n1 = int(utlf_a)
            max_n2 = int(utlf_b + utlf_d)
            min_n1 = 0
            min_n2 = 0

        c = 0

        t_base_change = np.linalg.inv(np.column_stack([self.t1, self.t2]))

        # base points of the cells accepted so far, i.e. the coordinates without the base vectors
        base_points = []

        # try n1, n2 such that n1*a1 + n2*a2 is in simulation torus
        for n2 in range(min_n2, max_n2):
            for n1 in range(min_n1, max_n1):
                # base point in lattice
                site_vec = n1 * self.a1 + n2 * self.a2

                # skip if an equivalent site is already there. Two sites are equivalent if they
                # differ by any torus vector, i.e. if the difference has integer coordinates in
                # the t basis. Comparing against a fixed list of shifts (+-t1, +-t2, ...) is not
                # enough: the accepted region below is |.| < 1 in both t coordinates, so sites on
                # opposite edges of it differ by 2*t1 or 2*t2 and were kept as separate sites.
                if len(base_points) > 0:
                    shifts = t_base_change @ (site_vec - np.array(base_points)).transpose()
                    if np.any(np.all(np.isclose(shifts, np.round(shifts)), axis=0)):
                        continue

                # stay in positive sector in t basis for both sublattice points
                vec_t_base = t_base_change @ site_vec

                # assert basis change worked
                assert (np.isclose(vec_t_base[0] * self.t1 + vec_t_base[1] * self.t2, site_vec).all())

                if not ((0 <= np.abs(vec_t_base[0]) < 1) and (0 <= np.abs(vec_t_base[1]) < 1)):
                    continue

                for c_b, base_vec in enumerate(self.base_vectors):
                    self.coordinates[c + c_b] = site_vec + base_vec
                c += c_b + 1
                base_points.append(site_vec)

                # if already enough sites found, assume rest are duplicates
                if c > self.n - 1:
                    break

            # the break above only leaves the inner loop, so stop the outer one as well
            if c > self.n - 1:
                break

        if c != self.n:
            print(c, max_n2 * max_n1 * len(self.base_vectors))
            print(self.n, id)
            raise AssertionError("site number not correct")

        # sort coordinates first by x component, then by y component
        # then by going through coordinates we go row by row upwards and in each row from left to right
        ind = np.lexsort((self.coordinates[:, 0], self.coordinates[:, 1]))
        self.coordinates = self.coordinates[ind]

        """get ids for sites"""
        # get y coordinates with the number of nodes which have it
        ys, y_count = np.unique(self.coordinates[:, 1], return_counts=True)
        x_id = 0
        y_id = 0
        i = 0

        while i < self.n:
            # go through each y-coordinate / row in lattice
            for row_el in range(y_count[y_id]):
                # assign id to list
                self.node_ids[i] = [y_id, x_id]

                # raise column index
                x_id += 1

                # raise counter
                i += 1

            # reset column index
            x_id = 0

            # raise row index
            y_id += 1

        # horizontal strings
        all_sites = list(np.arange(0, n))
        while(len(all_sites)>0):
            start_site = all_sites[0]
            neighbor = -1
            this_string = []
            curr_site = start_site
            while neighbor != start_site:
                neighbor = self.neighbor_r(curr_site)
                this_string.append(neighbor)
                try:
                    all_sites.remove(neighbor)
                except ValueError:
                    raise ValueError("Site belongs to several strings of the same type!!")
                curr_site = neighbor
            self.horizontal_strings.append(this_string)

        # up right strings
        all_sites = list(np.arange(0, n))
        while(len(all_sites)>0):
            start_site = all_sites[0]
            neighbor = -1
            this_string = []
            curr_site = start_site
            while neighbor != start_site:
                neighbor = self.neighbor_ur(curr_site)
                this_string.append(neighbor)
                try:
                    all_sites.remove(neighbor)
                except ValueError:
                    raise ValueError("Site belongs to several strings of the same type!!")
                curr_site = neighbor
            self.diagonal_strings_ur.append(this_string)

        # up left strings
        all_sites = list(np.arange(0, n))
        while(len(all_sites)>0):
            start_site = all_sites[0]
            neighbor = -1
            this_string = []
            curr_site = start_site
            while neighbor != start_site:
                neighbor = self.neighbor_ul(curr_site)
                this_string.append(neighbor)
                try:
                    all_sites.remove(neighbor)
                except ValueError:
                    raise ValueError("Site belongs to several strings of the same type!!")
                curr_site = neighbor
            self.diagonal_strings_ul.append(this_string)
        assert sorted([item for sublist in self.diagonal_strings_ul for item in sublist]) == list(range(n)), "Not all sites in loop"
        assert sorted([item for sublist in self.diagonal_strings_ur for item in sublist]) == list(range(n)), "Not all sites in loop"
        assert sorted([item for sublist in self.horizontal_strings for item in sublist]) == list(range(n)), "Not all sites in loop"
        return


    def neighbor_ind_to_ind(self, num):
        """
        6 neighbors of each triagonal lattice point in the order such that Majorana_Hubbard works as desired
        (counter-clockwise starting with bottom left)
        :param num: index of current site
        :return: list of integers that are the neighbors in counter-clockwise order starting with bottom left
        """

        # coordinates of current site
        my_coords = self.coordinates[num]

        # vectors that will be added to take periodic boundary conditions into account
        # Note: 0,0 is also part here to take the trivial case (neighbor in the middle of the lattice) into account
        torus_vec = [np.array([0, 0]).transpose(), self.t1, self.t2, -self.t1, -self.t2, self.t1 + self.t2, -1 * (self.t1 + self.t2),
                     self.t1 - self.t2, self.t2 - self.t1]

        # the six vectors of neighbor directions
        neighbor_directions = [(-1)*self.a2, s3@self.a2, self.a1, self.a2, (-1)*s3@self.a2, (-1)*self.a1]

        # list of neighbors
        neighbors = []

        # for each neighbor diretion
        for n_dir in neighbor_directions:

            # coordinate of the neighbor if its not at the boundary
            n_coord = my_coords + n_dir

            # determine if coordinate is a lattice site for the neighbor coordinate plus possible torus vector
            for t_vec in torus_vec:
                    try:
                        # Get index of coodinate vector in coordinate array
                        n1 = where_vec_array(self.coordinates, n_coord+t_vec)[0]
                    except IndexError:
                        continue
                    else:
                        # when successfull append to neighbor list
                        neighbors.append(n1)
                        break
            else:
                raise AssertionError("Neighbor not found")

        assert(len(neighbors) == 6)
        return neighbors

    def neighbor_ur(self, ind):
        return self.neighbor_ind_to_ind(ind)[3]

    def neighbor_ul(self, ind):
        return self.neighbor_ind_to_ind(ind)[4]

    def neighbor_r(self, ind):
        return self.neighbor_ind_to_ind(ind)[2]

    def calc_adjacency_matrix(self):
        """
        Calculates the adjacency matrix of the honeycomb graph
        :return:
        """
        for i in range(self.n):
            for j in range(self.n):
                if j in self.neighbor_ind_to_ind(i):
                    self.adjacencymatrix[i][j] = 1
                else:
                    self.adjacencymatrix[i][j] = 0

    def distance(self, i, j):
        ci = self.coordinates[i]
        cj = self.coordinates[j]
        return np.linalg.norm(ci - cj)

    def pacific_neighbors(self, i, j):
        """
        Given two Site i,j (integers) that are neighbors, return True if their connection is "on the other side",
        i.e. due to periodic boundary conditions
        :param i: integer in 0... n-1
        :param j: integer in 0... n-1
        :return: bool
        """
        if self.distance(i, j) > self.a * 1.005:
            return True
        else:
            return False

    def simple_cycles(self, i):
        if self.adjacencymatrix[0][0] == -1:
            self.calc_adjacency_matrix()

        # create graph
        G = nx.Graph()

        # go through all sites
        for i in range(self.n):
            maj_label = lambda v: int(np.where(self.new_order == v)[0])

            # add nodes
            G.add_node(tuple(self.node_ids[i]), pos=tuple(self.coordinates[i]), node_label= maj_label(i))

            # add edges
            for j in range(i + 1, self.n):
                if self.adjacencymatrix[i][j] == 1:
                    if self.pacific_neighbors(i, j):
                        G.add_edge(tuple(self.node_ids[i]), tuple(self.node_ids[j]))
            return nx.all_simple_paths(G, source=i, target=i)


    def plot(self, save_fig=False, highlight = None, highlight_triangles = None):
        """
        Creates and draws networkx graph
        :return:
        """
        # if not calculated so far, calc adjacency matrix
        if self.adjacencymatrix[0][0] == -1:
            self.calc_adjacency_matrix()

        # create graph
        G = nx.Graph()
        color = "dodgerblue"

        fig = plt.figure()
        fig.set_tight_layout(False)

        # go through all sites
        for i in range(self.n):
            maj_label = lambda v: int(np.where(self.new_order == v)[0])

            # add nodes
            G.add_node(tuple(self.node_ids[i]), pos=tuple(self.coordinates[i]), col=color, node_label= maj_label(i))

            # add edges
            for j in range(i + 1, self.n):
                if self.adjacencymatrix[i][j] == 1:
                    if self.pacific_neighbors(i, j):
                        st = "dotted"
                    else:
                        st = "solid"
                    # vertical link?
                    if self.coordinates[i][0] == self.coordinates[j][0]:
                        edge_label = int(maj_label(i)/2)
                    else:
                        edge_label = ""
                        pass

                    G.add_edge(tuple(self.node_ids[i]), tuple(self.node_ids[j]), style=st, label=edge_label)

        pos = nx.get_node_attributes(G, 'pos')

        colors = nx.get_node_attributes(G, 'col')
        color_list = [colors[idd] for idd in G.nodes]
        if highlight is not None:
            for el in highlight:
                color_list[el] = 'orange'
        node_labels = nx.get_node_attributes(G, 'node_label')
        # index_list = [indices[idd] for idd in G.nodes]

        styles = nx.get_edge_attributes(G, 'style')
        style_list = [styles[idd] for idd in G.edges]

        labels = nx.get_edge_attributes(G, 'label')
        label_list = [labels[idd] for idd in G.edges]
        print(labels)
        nx.draw_networkx_edges(G, pos, style=style_list)
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels, rotate=False)
        nx.draw_networkx_nodes(G, pos, node_color=color_list)

        # Choose here which labels to draw
        nx.draw_networkx_labels(G, pos, node_labels)

        plt.arrow(0, 0, self.t1[0], self.t1[1], width=0.01, head_width=0.15, head_length=0.2, color="green", alpha=1.0)
        plt.arrow(0, 0, self.t2[0], self.t2[1], width=0.01, head_width=0.15, head_length=0.2, color="green", alpha=1.0)
        #custom_lines = [Line2D([0], [0], color="white", marker='o', markerfacecolor="dodgerblue"),
        #                Line2D([0], [0], color="white", marker='o', markerfacecolor="orange")]
        #plt.legend(custom_lines, ["Sublattice A", "Sublattice B"])
        if highlight_triangles is not None:
            for tri in highlight_triangles:
                xs = [self.coordinates[i][0] for i in tri]
                ys = [self.coordinates[i][1] for i in tri]
                plt.fill(xs, ys, 'orange')
        if save_fig:
            plt.savefig("Lattices/Triangular/{}/Triangular{}.png".format(self.n, self.id))
        plt.show()
        return

    def plot_extended(self, num_x, num_y, save_fig=False, save_name=None,
                      highlight_sites = None, highlight_sites_colors=None,
                      highlight_triangles = None, highlight_triangles_colors = None,
                      lowlight_triangles=None, intersection_color='red',
                      legend=False, labels=None, title=None,
                      dont_show=False, img_path=None):

        """
                Creates and draws networkx graph
                :return:
                """
        # if not calculated so far, calc adjacency matrix
        if self.adjacencymatrix[0][0] == -1:
            self.calc_adjacency_matrix()

        # create graph
        G = nx.Graph()
        color = "dodgerblue"

        fig = plt.figure()
        fig.set_tight_layout(False)

        # go through all sites
        for i in range(self.n):
            for i_x in range(num_x):
                for i_y in range(num_y):

                    # add nodes
                    p = self.coordinates[i] + i_x*self.t1 + i_y*self.t2
                    G.add_node(tuple([i, i_x, i_y]), pos=tuple(p), col=color, node_label=i)

        pos = nx.get_node_attributes(G, 'pos')

        for c1, (k1, v1) in enumerate(pos.items()):
            for c2, (k2, v2) in enumerate(pos.items()):
                if c1<=c2:
                    continue

                if np.linalg.norm(np.array(v1) - np.array(v2)) < self.a + 0.001:
                    G.add_edge(k1, k2, style="solid")

        colors = nx.get_node_attributes(G, 'col')
        color_list = [colors[idd] for idd in G.nodes]

        node_labels = nx.get_node_attributes(G, 'node_label')
        # index_list = [indices[idd] for idd in G.nodes]

        styles = nx.get_edge_attributes(G, 'style')
        style_list = [styles[idd] for idd in G.edges]

        #labels = nx.get_edge_attributes(G, 'label')
        label_list = ["" for idd in G.edges]
        nx.draw_networkx_edges(G, pos)
        #nx.draw_networkx_edge_labels(G, pos=pos,  rotate=False, label_list)

        if highlight_sites is not None:

            assert(len(highlight_sites) == len(highlight_sites_colors))
            assert('red' not in highlight_sites_colors)

            for highlight_set, set_color in zip(highlight_sites, highlight_sites_colors):
                for h_site in highlight_set:
                    for i_x in range(num_x):
                        for i_y in range(num_y):
                            ind = list(G.nodes).index((h_site, i_x, i_y))
                            color_list[ind] = set_color
            try:
                intersection = list(np.intersect1d(*highlight_sites))
                if len(intersection) > 0:
                    for h_site in intersection:
                        for i_x in range(num_x):
                            for i_y in range(num_y):
                                ind = list(G.nodes).index((h_site, i_x, i_y))
                                color_list[ind] = 'red'
            except TypeError:
                pass


        nx.draw_networkx_nodes(G, pos, node_color=color_list)

        # Choose here which labels to draw
        nx.draw_networkx_labels(G, pos, node_labels)

        plt.arrow(self.coordinates[0,0], self.coordinates[0,1], self.t1[0], self.t1[1], width=0.01, head_width=0.15, head_length=0.2, color="green", alpha=1.0)
        plt.arrow(self.coordinates[0,0], self.coordinates[0,1], self.t2[0], self.t2[1], width=0.01, head_width=0.15, head_length=0.2, color="green", alpha=1.0)

        if highlight_triangles is not None:
            for h_tri_set, h_tri_col in zip(highlight_triangles, highlight_triangles_colors):

                # for triangle in highlight triangle set
                for tri in h_tri_set:

                    tri_nodes = []
                    for c_tri, tri_site in enumerate(tri):

                        # list for each site
                        tri_nodes.append([])

                        for node_tuple in G.nodes:
                            if node_tuple[0] == tri_site:
                                tri_nodes[c_tri].append(node_tuple)

                    for site1 in tri_nodes[0]:
                        for site2 in tri_nodes[1]:
                            for site3 in tri_nodes[2]:
                                p1 = np.array(pos[site1])
                                p2 = np.array(pos[site2])
                                p3 = np.array(pos[site3])
                                if np.linalg.norm(p1 - p2) < self.a + 0.001:
                                    if np.linalg.norm(p3 - p2) < self.a + 0.001:
                                        if np.linalg.norm(p1-p3) < self.a + 0.001:
                                            # assert only downpointing!
                                            if p1[1] < p2[1] and p1[1] < p3[1]:
                                                xs = [p1[0], p2[0], p3[0]]
                                                ys = [p1[1], p2[1], p3[1]]

                                                plt.fill(xs, ys, h_tri_col)

            # intersections = same triangles
            if len(highlight_triangles) == 2 and isinstance(highlight_triangles[0][0], list):
                intersections = []

                for tri in highlight_triangles[0]:
                    if tri in highlight_triangles[1]:
                        intersections.append(tri)

                for tri in intersections:
                    tri_nodes = []
                    for c_tri, tri_site in enumerate(tri):

                        # list for each site
                        tri_nodes.append([])

                        for node_tuple in G.nodes:
                            if node_tuple[0] == tri_site:
                                tri_nodes[c_tri].append(node_tuple)

                    for site1 in tri_nodes[0]:
                        for site2 in tri_nodes[1]:
                            for site3 in tri_nodes[2]:
                                p1 = np.array(pos[site1])
                                p2 = np.array(pos[site2])
                                p3 = np.array(pos[site3])
                                if np.linalg.norm(p1 - p2) < self.a + 0.001:
                                    if np.linalg.norm(p3 - p2) < self.a + 0.001:
                                        if np.linalg.norm(p1-p3) < self.a + 0.001:
                                            if p1[1] < p2[1] and p1[1] < p3[1]:
                                                xs = [p1[0], p2[0], p3[0]]
                                                ys = [p1[1], p2[1], p3[1]]
                                                plt.fill(xs, ys, intersection_color)

            elif len(highlight_triangles) > 2:
                raise ValueError('Cannot deal with three set of highlight triangles so far')


        if lowlight_triangles is not None:
            for tri in lowlight_triangles:
                tri_nodes = []
                for c_tri, tri_site in enumerate(tri):

                    # list for each site
                    tri_nodes.append([])

                    for node_tuple in G.nodes:
                        if node_tuple[0] == tri_site:
                            tri_nodes[c_tri].append(node_tuple)

                for site1 in tri_nodes[0]:
                    for site2 in tri_nodes[1]:
                        for site3 in tri_nodes[2]:
                            p1 = np.array(pos[site1])
                            p2 = np.array(pos[site2])
                            p3 = np.array(pos[site3])
                            if np.linalg.norm(p1 - p2) < self.a + 0.001:
                                if np.linalg.norm(p3 - p2) < self.a + 0.001:
                                    if np.linalg.norm(p1-p3) < self.a + 0.001:
                                        if p1[1] < p2[1] and p1[1] < p3[1]:
                                            xs = [p1[0], p2[0], p3[0]]
                                            ys = [p1[1], p2[1], p3[1]]
                                            plt.fill(xs, ys, 'silver')
        if title is not None:
            plt.title(title)
        if legend:
            bg_col = mpl.rcParams["axes.facecolor"]
            custom_lines = [Line2D([0], [0], marker='v', color=bg_col, markerfacecolor=c, markersize=15) for c in highlight_triangles_colors]
            custom_lines.append(Line2D([0], [0], marker='v', color=bg_col, markerfacecolor=intersection_color, markersize=15))
            custom_lines.append(Line2D([0], [0], marker='o', color=bg_col , label='Scatter',
                          markerfacecolor=highlight_sites_colors[0], markersize=15))
            plt.legend(custom_lines, labels, loc=2, fontsize='xx-large')

        if save_fig:
            if save_name is None:
                raise ValueError
            save_name = f"n{self.n}_ID{self.id}_{save_name}"
            # savefig(save_name, meta_description="", timestamp=False, destination=img_path)
            plt.savefig(save_name, format="pdf")
        if not dont_show:
            plt.show()
        else:
            plt.close()
        return



    def torus_coordinates(self, x, y, R=6, r=4, in_tbasis=False):
        '''
        in_tbasis: False:
            Given any (x,y) coordinates of a point in the plot, return x,y,z the cartesian coordinates of the corresponding 3D torus points
        in_tbasis: True:
        Given x,y in [0,1) the components of any vector in the lattice in simulation torus basis
            return x,y,z the cartesian coordinates of the corresponding 3D torus point
        :param x:
        :param y:
        :param R:
        :param r:
        :param in_tbasis:
        :return:
        '''

        if not in_tbasis:
        # components in terms of torus simulation vectors
            inv = np.linalg.inv(np.column_stack([self.t1, self.t2]))
            cx = inv[0,0] * x + inv[0,1] * y
            cy = inv[1,0] * x + inv[1,1] * y
        else:
            cx = x
            cy = y

        # angles
        phi = 2* np.pi*cx
        theta = 2* np.pi * cy

        # cartesian coordinates on the torus surface
        x = (R + r * np.cos(theta)) * np.cos(phi)
        y = (R + r * np.cos(theta)) * np.sin(phi)
        z = (r * np.sin(theta))
        return x, y, z

    def plot_on_torus(self, save_fig=False, lowlight_triangles=None):
        """
            Creates and draws networkx graph
            :return:
            """
        # if not calculated so far, calc adjacency matrix
        if self.adjacencymatrix[0][0] == -1:
            self.calc_adjacency_matrix()

        # create graph
        G = nx.Graph()

        fig = plt.figure()
        ax = Axes3D(fig)
        fig.set_tight_layout(False)

        # go through all sites
        for i in range(self.n):


            # add nodes
            coordinates_torus = self.torus_coordinates(*self.coordinates[i])
            G.add_node(i, pos=tuple(coordinates_torus), col="dodgerblue", node_label=i)

            # add edges
            for j in range(i + 1, self.n):
                if self.adjacencymatrix[i][j] == 1:
                    if self.pacific_neighbors(i, j):
                        st = "dotted"
                    else:
                        st = "solid"

                    G.add_edge(i, j, style=st, label="")

        pos = nx.get_node_attributes(G, 'pos')

        colors = nx.get_node_attributes(G, 'col')
        color_list = [colors[idd] for idd in G.nodes]

        node_labels = nx.get_node_attributes(G, 'node_label')
        # index_list = [indices[idd] for idd in G.nodes]

        styles = nx.get_edge_attributes(G, 'style')
        style_list = [styles[idd] for idd in G.edges]

        labels = nx.get_edge_attributes(G, 'label')
        label_list = [labels[idd] for idd in G.edges]

        #nx.draw_networkx_edges(G, pos, style=style_list)
        #nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels, rotate=False)
        #nx.draw_networkx_nodes(G, pos, node_color=color_list)

        # Choose here which labels to draw
        #nx.draw_networkx_labels(G, pos, node_labels)

        #plt.arrow(0, 0, self.t1[0], self.t1[1], width=0.01, head_width=0.15, head_length=0.2, color="green", alpha=1.0)
        #plt.arrow(0, 0, self.t2[0], self.t2[1], width=0.01, head_width=0.15, head_length=0.2, color="green", alpha=1.0)

        # plot the torus itself:
        xs = np.linspace(0, 1, 50)
        ys = np.linspace(0, 1, 50)
        X, Y = np.meshgrid(xs, ys)

        ax.plot_surface(*self.torus_coordinates(X, Y, in_tbasis=True), alpha=0.4, color='blue')

        for key, values in pos.items():
            ax.scatter(values[0], values[1], values[2], color='red', s=20)
            ax.text(*values, key)

        for i,j, edge_props in G.edges(data=True):
            if edge_props['style'] == "solid":
                xs = np.linspace(self.coordinates[i][0], self.coordinates[j][0], 25)
                ys = np.linspace(self.coordinates[i][1], self.coordinates[j][1], 25)
                ax.plot(*self.torus_coordinates(xs, ys), color="black")
            else:
                i_xy = np.array([self.coordinates[i][0], self.coordinates[i][1]])
                j_xy = np.array([self.coordinates[j][0], self.coordinates[j][1]])

                for t_vec in [self.t1, self.t2, -self.t1, -self.t2, self.t1 + self.t2, -1 * (self.t1 + self.t2),
                     self.t1 - self.t2, self.t2 - self.t1]:
                    if np.linalg.norm((i_xy+t_vec)-j_xy) < self.a + 0.01:
                        inv = np.linalg.inv(np.column_stack([self.t1, self.t2]))
                        i_cx = inv[0, 0] * self.coordinates[i][0] + inv[0, 1] * self.coordinates[i][1]
                        i_cy = inv[1, 0] * self.coordinates[i][0] + inv[1, 1] * self.coordinates[i][1]
                        j_cx = inv[0, 0] * self.coordinates[j][0] + inv[0, 1] * self.coordinates[j][1]
                        j_cy = inv[1, 0] * self.coordinates[j][0] + inv[1, 1] * self.coordinates[j][1]

                        if np.allclose(t_vec, self.t1):
                            t1_c = np.linspace(i_cx, j_cx+1)%1
                            t2_c = np.linspace(i_cy, j_cy)
                        elif np.allclose(t_vec, -self.t1):
                            t1_c = np.linspace(i_cx+1, j_cx)%1
                            t2_c = np.linspace(i_cy, j_cy)
                        elif np.allclose(t_vec, self.t2):
                            t1_c = np.linspace(i_cx, j_cx)
                            t2_c = np.linspace(i_cy, j_cy+1)%1
                        elif np.allclose(t_vec, -self.t2):
                            t1_c = np.linspace(i_cx, j_cx)
                            t2_c = np.linspace(i_cy+1, j_cy)%1
                        elif np.allclose(t_vec, self.t1 + self.t2):
                            t1_c = np.linspace(i_cx, j_cx+1)%1
                            t2_c = np.linspace(i_cy, j_cy+1)%1
                        elif np.allclose(t_vec, -self.t1 + self.t2):
                            t1_c = np.linspace(i_cx+1, j_cx)%1
                            t2_c = np.linspace(i_cy, j_cy+1)%1
                        elif np.allclose(t_vec, -self.t1 - self.t2):
                            t1_c = np.linspace(i_cx+1, j_cx)%1
                            t2_c = np.linspace(i_cy+1, j_cy)%1
                        elif np.allclose(t_vec, self.t1 - self.t2):
                            t1_c = np.linspace(i_cx, j_cx+1)%1
                            t2_c = np.linspace(i_cy+1, j_cy)%1
                        else:
                            raise ValueError

                        ax.plot(*self.torus_coordinates(t1_c, t2_c, in_tbasis=True), color='black', linestyle='dashed')

        if save_fig:
            plt.savefig("Lattices/Honeycomb/{}/Honeycomb{}_torus.png".format(self.n, self.id))
        plt.show()
        return




    def draw_basic_network(self, fig, ax):
        """
        Creates and draws networkx graph
        :param highlight: indices 0...n that will be plotted in another color
        :return:
        """
        # if not calculated so far, calc adjacency matrix
        if self.adjacencymatrix[0][0] == -1:
            self.calc_adjacency_matrix()

        # create graph
        G = x.Graph()

        # fig = plt.figure()
        # fig.set_tight_layout(False)

        # go through all sites
        for i in range(self.n):

            color = "dodgerblue"

            # add nodes
            G.add_node(tuple(self.node_ids[i]), pos=tuple(self.coordinates[i]), col=color)

            # add edges
            for j in range(i + 1, self.n):
                if self.adjacencymatrix[i][j] == 1:
                    if self.pacific_neighbors(i, j):
                        st = "dotted"
                    else:
                        st = "solid"
                    G.add_edge(tuple(self.node_ids[i]), tuple(self.node_ids[j]), style=st)

        pos = nx.get_node_attributes(G, 'pos')
        colors = nx.get_node_attributes(G, 'col')
        color_list = [colors[idd] for idd in G.nodes]
        styles = nx.get_edge_attributes(G, 'style')
        style_list = [styles[idd] for idd in G.edges]
        nx.draw(G, pos, ax=ax, node_color=color_list, style=style_list, with_labels=False)
        ax.arrow(0, 0, self.t1[0], self.t1[1], width=0.01, head_width=0.15, head_length=0.2, color="green", alpha=1.0)
        ax.arrow(0, 0, self.t2[0], self.t2[1], width=0.01, head_width=0.15, head_length=0.2, color="green", alpha=1.0)
        return

    def draw_highlights(self, highlights, fig, ax, lowlight_color="dodgerblue", highlight_color="orange", with_labels=True):
        new_axis = fig.add_axes(ax)
        G = nx.Graph()
        for i in range(self.n):

            # # if site in highlight list
            if i in highlights:
                color = highlight_color
                if with_labels:
                    label = highlights.index(i)
                else:
                    label = ""
            else:
                color = lowlight_color
                label = " "

            # add nodes
            G.add_node(tuple(self.node_ids[i]), pos=tuple(self.coordinates[i]), col=color, label=label)

        pos = nx.get_node_attributes(G, 'pos')
        colors = nx.get_node_attributes(G, 'col')
        labels = nx.get_node_attributes(G, "label")
        color_list = [colors[idd] for idd in G.nodes]
        nx.draw(G, pos, ax=new_axis, node_color=color_list, labels=labels)
        return new_axis



# ---------------------------------------------------------------------------------------------------------------------
class Honeycomb:
    """
    Create a Honeycomb Lattice as in the Lattice Catalogue (LC)
    Given an id as in the lattice catalogue, this function calculates the coordinates of the corresponing lattice
    as well as the utlf matrix, the simulation torus
    """

    def __init__(self, n, id, order="paper"):
        """

        :param n: Number of sites
        :param id: Lattice ID
        """

        # number of sites
        self.n = n

        # lattice constant
        self.a = 2. / (np.sqrt(3))

        # lattice vectors
        self.a1 = np.array([np.sqrt(3) * self.a / 2., 3. * self.a / 2]).transpose()
        self.a2 = np.array([np.sqrt(3.) * self.a, 0]).transpose()

        # Lattice ID as in LC
        # fill utlf matrix from lattice id
        self.id = str(id)
        utlf_d = int(self.id[-2::])
        utlf_b = int(self.id[-4::][:2])
        utlf_a = int(self.id[:len(self.id) - 4])

        # UTLF Matrix, Simulation Torus as in LC
        self.utlf = np.zeros(shape=(2, 2), dtype=float)


        assert utlf_b < utlf_d  # see lattice catalogue pdf
        self.utlf = np.array([[utlf_a, utlf_b], [0, utlf_d]], dtype=float)

        self.simulation_torus = np.zeros(shape=(2, 2), dtype=float)
        # get simulation torus from utlf matrix
        self.simulation_torus = np.column_stack([utlf_a * self.a1 + utlf_b * self.a2, utlf_d * self.a2])

        # (x,y) coordinates for each site with site 0 at (0,0) -> therefore with -1 initialized because
        # we check at some point if coordinates already in array
        self.coordinates = np.full((self.n, 2), -1.)

        # node ids which will be tuple (row_index, column_index)
        self.node_ids = np.zeros(shape=(self.n, 2), dtype=int)

        # adjacency matrix of graph
        self.adjacencymatrix = np.full((self.n, self.n), -1)

        # order of indices of lattice sites
        self.order = order
        # order list, will be a mapping index->value thus default is index=value
        self.new_order = np.arange(self.n)
        """
        Note:
            for a site i with coordinates self.coordinates[i] the corresponding Majorana label 
            is np.where(self.new_order = i)
        """

        # Indices of the central nodes that are on the same vertical/horizontal string on the triangular lattice

        self.vertical_strings = []
        self.horizontal_strings = []

        # t_1 and t_2 vectors for lattices where the vectors are not the components of the utlf matrix but "geometrically equivalent"
        # n: {Majorana_Number: {Lattice ID: [[t1[0] t1[0]],[[t1[0], t2[1]]}}
        self.sim_tor_dict = {6: {10103: [[1, 1], [1, -2]]},
                             8: {10104: [[1, 1], [2, -2]]},
                             10: {10105: [[1, 1], [2, -3]]},
                             12: {10106: [[1, 1], [3, -3]],
                                  10206: [[1, 2], [2, -2]]},
                             14: {10207: [[1, 2], [3, -1]],
                                  10107: [[1, 1], [3, -4]]},
                             16: {10208: [[1, 2], [3, -2]],
                                  10308: [[1, 3], [2, -2]],
                                  10108: [[1, 1], [4, -4]]},
                             18: {10109: [[1, 1], [4, -5]],
                                  10209: [[1, 2], [4, -1]]},
                             20: {10110: [[1, 1], [5, -5]],
                                  10210: [[1, 2], [4, -2]]},
                             22: {10111: [[1, 1], [5, -6]],
                                  10211: [[1, 2], [4, -3]]},
                             24: {20206: [[2, 2], [2, -4]]},
                             26: {10313: [[1, 3], [4, -1]],
                                  10113: [[1, 1], [5, -8]]
                                  },
                             28: {10114: [[1, 1], [5, -9]],
                                  10314: [[1, 3], [4, -2]]},
                             30: {10315: [[1, 3], [4, -3]],
                                  10415: [[1, 4], [4, 1]]},
                             32: {20208: [[2, 2], [4, -4]],
                                  10616: [[2, -4],[3, 2]],
                                  10716: [[2, -2], [3, 5]],
                                  10116: [[1,1],[5, -11]],
                                  10316: [[1,3],[5, -1]],
                                  10216: [[1, 2], [5, -6]]},
                             34: {10117: [[1, 1], [5, -12]],
                                  10317: [[1,3],[5, -2]],
                                  10217: [[1,2],[5, -7]]},
                             36: {10118: [[1,1],[5, -13]],
                                  10418: [[1,4],[4, -2]],
                                  10318: [[1,3], [5, -3]],
                                  10518: [[3, -3], [4, 2]],
                                  10218: [[1,2], [5, -8]],
                                  10818: [[2, -2],[5, 4]]}}


        # either get t1, t2 vectors from simulation torus or from the dicitonary
        try:
            mat = self.sim_tor_dict[self.n][int(self.id)]

            t1_comp = np.array(mat[0]).transpose()
            t2_comp = np.array(mat[1]).transpose()
            # print("dict used instead of utlf matrix for t_1 and t_2 vectors")

            l0 = [int(el) for el in [0., t1_comp[0], t2_comp[0], t1_comp[0] + t2_comp[0]]]
            max_n1 = np.amax(l0)
            min_n1 = np.amin(l0)

            l1 = [int(el) for el in [0., t1_comp[1], t2_comp[1], t1_comp[1] + t2_comp[1]]]
            max_n2 = np.amax(l1)
            min_n2 = np.amin(l1)

            self.t1 = t1_comp[0] * self.a1 + t1_comp[1] * self.a2
            self.t2 = t2_comp[0] * self.a1 + t2_comp[1] * self.a2

        except KeyError:
            self.t1 = self.simulation_torus[:, 0]
            self.t2 = self.simulation_torus[:, 1]

            max_n1 = int(utlf_a)
            max_n2 = int(utlf_b + utlf_d)
            min_n1 = 0
            min_n2 = 0

        c = 0

        torus_vec = [self.t1, self.t2, -self.t1, -self.t2, self.t1 + self.t2, -1 * (self.t1 + self.t2),
                     self.t1 - self.t2, self.t2 - self.t1]

        # try n1, n2 such that n1*a1 + n2*a2 is in simulation torus
        for n2 in range(min_n2, max_n2):
            for n1 in range(min_n1, max_n1):
                # base point in lattice
                site_vec = n1 * self.a1 + n2 * self.a2

                # second sublattice point
                site_vec_2 = site_vec + np.array([0, self.a]).transpose()

                # avoid duplicates
                if len(where_vec_array(self.coordinates, site_vec)) > 0:
                    continue

                # check other site of the torus
                duplicates = []
                for periodic_vec in torus_vec:
                    if len(where_vec_array(self.coordinates, site_vec + periodic_vec)) > 0:
                        duplicates.append(1)
                if len(duplicates) > 0:
                    continue

                # stay in positive sector in t basis for both sublattice points
                vec_t_base = np.linalg.inv(np.column_stack([self.t1, self.t2])) @ site_vec
                vec_t_base_2 = np.linalg.inv(np.column_stack([self.t1, self.t2])) @ site_vec_2

                # assert basis change worked
                assert (np.isclose(vec_t_base[0] * self.t1 + vec_t_base[1] * self.t2, site_vec).all())
                assert (np.isclose(vec_t_base_2[0] * self.t1 + vec_t_base_2[1] * self.t2, site_vec_2).all())

                if not ((0 <= vec_t_base[0] < 1) and (0 <= vec_t_base[1] < 1)):
                    continue

                # TODO implement that second sublattice point is also in positive t sector...
                # if not ((0 <= vec_t_base_2[0] < 1) and (0 <= vec_t_base_2[1] < 1)):
                #    continue

                self.coordinates[c] = site_vec
                self.coordinates[c + 1] = site_vec + np.array([0, self.a]).transpose()
                c += 2

                # if already enough sites found, assume rest are duplicates
                if c > self.n - 1:
                    break

        if c != self.n:
            print("Lattice", self.id)
            print(c, max_n2 * max_n1 * 2)
            raise AssertionError("site number not correct")

        # sort coordinates first by x component, then by y component
        # then by going through coordinates we go row by row upwards and in each row from left to right
        ind = np.lexsort((self.coordinates[:, 0], self.coordinates[:, 1]))
        self.coordinates = self.coordinates[ind]

        """get ids for sites"""
        # get y coordinates with the number of nodes which have it
        ys, y_count = np.unique(self.coordinates[:, 1], return_counts=True)
        xs, x_count = np.unique(self.coordinates[:, 0], return_counts=True)

        x_id = 0
        y_id = 0
        i = 0

        while i < self.n:

            self.horizontal_strings.append([])

            # go through each y-coordinate / row in lattice
            for row_el in range(y_count[y_id]):
                # assign id to list
                self.node_ids[i] = [y_id, x_id]
                self.horizontal_strings[y_id].append(i)
                # raise column index
                x_id += 1

                # raise counter
                i += 1

            # reset column index
            x_id = 0

            # raise row index
            y_id += 1

        """The following is only for the vertical strings"""
        x_id = 0
        y_id = 0
        i = 0

        while i < self.n:

            self.vertical_strings.append([])

            # go through each y-coordinate / row in lattice
            for col_el in range(x_count[x_id]):
                self.vertical_strings[x_id].append(i)
                # raise row index
                y_id += 1

                # raise counter
                i += 1

            # reset row index
            y_id = 0

            # raise col index
            x_id += 1

        """While the index in all the arrays so far is just 'arbitrary' the 'new_order" array contains the mapping onto
        the numbering which is also used for the mapping of Majoranas."""

        if self.order == "paper":   # use same order as in the paper

            x_id = 0
            y_id = 0
            i = 0

            while i < self.n:
                try:
                    a_site = where_vec_array(self.node_ids, [y_id + 1, x_id])[0]
                    b_site = where_vec_array(self.node_ids, [y_id, x_id])[0]
                except IndexError:
                    x_id = 0
                    y_id += 2
                    a_site = where_vec_array(self.node_ids, [y_id + 1, x_id])[0]
                    b_site = where_vec_array(self.node_ids, [y_id, x_id])[0]

                self.new_order[i] = a_site
                self.new_order[i + 1] = b_site
                x_id += 1
                i += 2
        return

    def neighbor_ind_to_ind(self, num):
        """
        Given the index (0...n-1) of a site, return the indices of it 3 neighbors in the order that is demanded by the Majorana-Hubbard Interaction Hamiltonians
        :param num: integer
        :return: list of three integers corresponding to the neighbors
        """

        # list that will be returned
        neighbor_inds = []

        # Vector from the upper site of a vertical bond to the upright - neighbor
        diag_vec_upright = self.a1 - np.array([0, self.a]).transpose()

        # Vector from the lower site of a vertical bond to the downright - neighbor
        diag_vec_downright = np.array([diag_vec_upright[0], -1 * diag_vec_upright[1]])

        # Vectors around the torus and their sums/differences
        # tries first the single vectors, then their sum and differences as the latter vectors may twist the lattice? # TODO check Twisting!!?!
        torus_vec = [self.t1, self.t2, -self.t1, -self.t2, self.t1 + self.t2, -1 * (self.t1 + self.t2),
                     self.t1 - self.t2, self.t2 - self.t1]

        # if current node is
        if self.node_ids[num][0] % 2 == 0:  # lower site in unit cell

            # add neighbor:
            # site above (number 2 in 4 majorana term in H_int)

            # its coordinate
            c1 = self.coordinates[num] + np.array([0, self.a]).transpose()

            # its index
            ind1 = where_vec_array(self.coordinates, c1)

            # append index to return list
            neighbor_inds.append(ind1[0])

            # 2nd neighbor:
            # site down left: (number 3 in 4majorana term)
            c2 = self.coordinates[num] - diag_vec_upright
            ind2 = where_vec_array(self.coordinates, c2)

            # if neighbor is on the other side of the "2D torus map/picture" (like in an atlas...)
            if len(ind2) == 0:

                # try going around the torus with the torus vectors
                for c_vec, vec in enumerate(torus_vec):
                    c2 = self.coordinates[num] - diag_vec_upright + vec
                    ind2 = where_vec_array(self.coordinates, c2)
                    if len(ind2) == 1:
                        break

                else:  # "if not break in the for loop"
                    print(ind2, self.coordinates[num] + self.t1 + self.t2)
                    print(self.coordinates)
                    raise AssertionError("neighbor 2 not found")
            neighbor_inds.append(ind2[0])

            # third neighbor
            # site down right: (number 4 in majorana term)
            c3 = self.coordinates[num] + diag_vec_downright
            ind3 = where_vec_array(self.coordinates, c3)
            if len(ind3) != 1:
                for c_vec, vec in enumerate(torus_vec):
                    c3 = self.coordinates[num] + diag_vec_downright + vec
                    ind3 = where_vec_array(self.coordinates, c3)
                    if len(ind3) == 1:
                        break
                else:
                    print(ind3)
                    raise AssertionError("neighbor 3 not found")
            neighbor_inds.append(ind3[0])

        elif self.node_ids[num][0] % 2 == 1:  # upper site in unit cell

            # first neighbor, up left
            c3 = self.coordinates[num] - diag_vec_downright
            ind3 = where_vec_array(self.coordinates, c3)
            if len(ind3) != 1:
                for c_vec, vec in enumerate(torus_vec):
                    c3 = self.coordinates[num] - diag_vec_downright + vec
                    ind3 = where_vec_array(self.coordinates, c3)
                    if len(ind3) == 1:
                        break
                else:
                    print(ind3)
                    raise AssertionError("neighbor 3 not found")
            assert (len(ind3) == 1)
            neighbor_inds.append(ind3[0])

            # second neighbor, up right
            c2 = self.coordinates[num] + diag_vec_upright
            ind2 = where_vec_array(self.coordinates, c2)
            if len(ind2) == 0:
                for c_vec, vec in enumerate(torus_vec):
                    c2 = self.coordinates[num] + diag_vec_upright + vec
                    ind2 = where_vec_array(self.coordinates, c2)
                    if len(ind2) == 1:
                        break
                else:
                    print(ind2)
                    raise AssertionError("neighbor 2 not found")
            assert (len(ind2) == 1)
            neighbor_inds.append(ind2[0])

            # 3rd neighbor, below
            c1 = self.coordinates[num] - np.array([0, self.a]).transpose()
            ind1 = where_vec_array(self.coordinates, c1)
            if len(ind1) != 1:
                print("my coords {}, my neighbor isn't {}".format(self.coordinates[num], c1))
                print(self.coordinates)
            assert (len(ind1) == 1)
            neighbor_inds.append(ind1[0])

        if len(neighbor_inds) != 3:
            print(neighbor_inds)
            raise AssertionError("neighbor missing or too much")
        return neighbor_inds

    def calc_adjacency_matrix(self):
        """
        Calculates the adjacency matrix of the honeycomb graph
        :return:
        """
        for i in range(self.n):
            for j in range(self.n):
                if j in self.neighbor_ind_to_ind(i):
                    self.adjacencymatrix[i][j] = 1
                else:
                    self.adjacencymatrix[i][j] = 0

    def distance(self, i, j):
        ci = self.coordinates[i]
        cj = self.coordinates[j]
        return np.linalg.norm(ci - cj)

    def pacific_neighbors(self, i, j):
        """
        Given two Site i,j (integers) that are neighbors, return True if their connection is "on the other side",
        i.e. due to periodic boundary conditions
        :param i: integer in 0... n-1
        :param j: integer in 0... n-1
        :return: bool
        """
        if self.distance(i, j) > self.a * 1.05:
            return True
        else:
            return False


    def plot(self, save_fig=False):
        """
        Creates and draws networkx graph
        :return:
        """
        # if not calculated so far, calc adjacency matrix
        if self.adjacencymatrix[0][0] == -1:
            self.calc_adjacency_matrix()

        # create graph
        G = nx.Graph()

        fig = plt.figure()
        fig.set_tight_layout(False)

        # go through all sites
        for i in range(self.n):

            # # if site in highlight list
            if i in self.ind_of_sublattice(0):
                color = "orange"    #B LATTIVE
            else:
                color = "dodgerblue"

            maj_label = lambda v: int(np.where(self.new_order == v)[0])

            # add nodes
            G.add_node(tuple(self.node_ids[i]), pos=tuple(self.coordinates[i]), col=color, node_label= maj_label(i))

            # add edges
            for j in range(i + 1, self.n):
                if self.adjacencymatrix[i][j] == 1:
                    if self.pacific_neighbors(i, j):
                        st = "dotted"
                    else:
                        st = "solid"
                    # vertical link?
                    if self.coordinates[i][0] == self.coordinates[j][0]:
                        edge_label = int(maj_label(i)/2)
                    else:
                        edge_label = ""
                        pass

                    G.add_edge(tuple(self.node_ids[i]), tuple(self.node_ids[j]), style=st, label=edge_label)

        pos = nx.get_node_attributes(G, 'pos')

        colors = nx.get_node_attributes(G, 'col')
        color_list = [colors[idd] for idd in G.nodes]

        node_labels = nx.get_node_attributes(G, 'node_label')
        # index_list = [indices[idd] for idd in G.nodes]

        styles = nx.get_edge_attributes(G, 'style')
        style_list = [styles[idd] for idd in G.edges]

        labels = nx.get_edge_attributes(G, 'label')
        label_list = [labels[idd] for idd in G.edges]
        print(labels)
        nx.draw_networkx_edges(G, pos, style=style_list)
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels, rotate=False)
        nx.draw_networkx_nodes(G, pos, node_color=color_list)

        # Choose here which labels to draw
        nx.draw_networkx_labels(G, pos, node_labels)

        plt.arrow(0, 0, self.t1[0], self.t1[1], width=0.01, head_width=0.15, head_length=0.2, color="green", alpha=1.0)
        plt.arrow(0, 0, self.t2[0], self.t2[1], width=0.01, head_width=0.15, head_length=0.2, color="green", alpha=1.0)
        custom_lines = [Line2D([0], [0], color="white", marker='o', markerfacecolor="dodgerblue"),
                        Line2D([0], [0], color="white", marker='o', markerfacecolor="orange")]
        plt.legend(custom_lines, ["Sublattice A", "Sublattice B"])

        if save_fig:
            plt.savefig("Lattices/Honeycomb/{}/Honeycomb{}.png".format(self.n, self.id))
        plt.show()
        return

    def draw_basic_network(self, fig, ax):
        """
        Creates and draws networkx graph
        :param highlight: indices 0...n that will be plotted in another color
        :return:
        """
        # if not calculated so far, calc adjacency matrix
        if self.adjacencymatrix[0][0] == -1:
            self.calc_adjacency_matrix()

        # create graph
        G = nx.Graph()

        # fig = plt.figure()
        # fig.set_tight_layout(False)

        # go through all sites
        for i in range(self.n):

            color = "dodgerblue"

            # add nodes
            G.add_node(tuple(self.node_ids[i]), pos=tuple(self.coordinates[i]), col=color)

            # add edges
            for j in range(i + 1, self.n):
                if self.adjacencymatrix[i][j] == 1:
                    if self.pacific_neighbors(i, j):
                        st = "dotted"
                    else:
                        st = "solid"
                    G.add_edge(tuple(self.node_ids[i]), tuple(self.node_ids[j]), style=st)

        pos = nx.get_node_attributes(G, 'pos')
        colors = nx.get_node_attributes(G, 'col')
        color_list = [colors[idd] for idd in G.nodes]
        styles = nx.get_edge_attributes(G, 'style')
        style_list = [styles[idd] for idd in G.edges]
        nx.draw(G, pos, ax=ax, node_color=color_list, style=style_list, with_labels=False)
        ax.arrow(0, 0, self.t1[0], self.t1[1], width=0.01, head_width=0.15, head_length=0.2, color="green", alpha=1.0)
        ax.arrow(0, 0, self.t2[0], self.t2[1], width=0.01, head_width=0.15, head_length=0.2, color="green", alpha=1.0)
        return

    def draw_highlights(self, highlights, fig, ax, lowlight_color="dodgerblue", highlight_color="orange", with_labels=True):
        new_axis = fig.add_axes(ax)
        G = nx.Graph()
        for i in range(self.n):

            # # if site in highlight list
            if i in highlights:
                color = highlight_color
                if with_labels:
                    label = highlights.index(i)
                else:
                    label = ""
            else:
                color = lowlight_color
                label = " "

            # add nodes
            G.add_node(tuple(self.node_ids[i]), pos=tuple(self.coordinates[i]), col=color, label=label)

        pos = nx.get_node_attributes(G, 'pos')
        colors = nx.get_node_attributes(G, 'col')
        labels = nx.get_node_attributes(G, "label")
        color_list = [colors[idd] for idd in G.nodes]
        nx.draw(G, pos, ax=new_axis, node_color=color_list, labels=labels)
        return new_axis

    def ind_of_sublattice(self, which):
        sublattice_sites = []
        for i in range(self.n):
            if self.node_ids[i][0] % 2 == which:
                sublattice_sites.append(i)
        assert (len(sublattice_sites) == int(self.n / 2))
        return sublattice_sites

# ---------------------------------------------------------------------------------------------------------------------
class Kagome:
    def __init__(self, n, id, order="paper"):
        """

        :param n: Number of sites
        :param id: Lattice ID
        """

        # number of sites
        self.n = n

        # lattice constant
        self.a = 1.

        # lattice vectors
        self.a1 = np.array([self.a, 0]).transpose()
        self.a2 = np.array([self.a/2., np.sqrt(3)*self.a/2]).transpose()
        self.base_vectors = [np.array([0., 0.]).transpose(),np.array([self.a/2., 0.]).transpose(),np.array([self.a/4., np.sqrt(3)*self.a/4]).transpose()]

        # Lattice ID as in LC
        # fill utlf matrix from lattice id
        if id != 1400014:
            self.id = str(id)
            utlf_d = int(self.id[-2::])
            utlf_b = int(self.id[-4::][:2])
            utlf_a = int(self.id[:len(self.id) - 4])
            # UTLF Matrix, Simulation Torus as in LC
            self.utlf = np.zeros(shape=(2, 2), dtype=float)
        else:
            self.id = str(1400014)
            utlf_d = 14.
            utlf_b = 0.
            utlf_a = 14.

        assert utlf_b < utlf_d  # see lattice catalogue pdf
        self.utlf = np.array([[utlf_a, utlf_b], [0, utlf_d]], dtype=float)

        self.simulation_torus = np.zeros(shape=(2, 2), dtype=float)
        # get simulation torus from utlf matrix
        self.simulation_torus = np.column_stack([utlf_a * self.a1 + utlf_b * self.a2, utlf_d * self.a2])

        # (x,y) coordinates for each site with site 0 at (0,0) -> therefore with -1 initialized because
        # we check at some point if coordinates already in array
        self.coordinates = np.full((self.n, 2), -1.)

        # node ids which will be tuple (row_index, column_index)
        self.node_ids = np.zeros(shape=(self.n, 2), dtype=int)

        # adjacency matrix of graph
        self.adjacencymatrix = np.full((self.n, self.n), -1)

        # order of indices of lattice sites
        self.order = order
        # order list, will be a mapping index->value thus default is index=value
        self.new_order = np.arange(self.n)
        """
        Note:
            for a site i with coordinates self.coordinates[i] the corresponding Majorana label 
            is np.where(self.new_order = i)
        """

        # list of sites in the horizontal direction
        self.horizontal_strings = []

        # in the "up-right" direction
        self.diagonal_strings_ur = []

        # in the "up-left" direction
        self.diagonal_strings_ul = []

        # t_1 and t_2 vectors for lattices where the vectors are not the components of the utlf matrix but "geometrically equivalent"
        # n: {Majorana_Number: {Lattice ID: [[t1[0] t1[0]],[[t1[0], t2[1]]}}
        self.sim_tor_dict = {4: {10104 : [[1, 1],[2, -2]]},
                             5: {10105: [[1, 1],[2, -3]]},
                             6: {10106: [[1,1], [3, -3]],
                                 10206: [[1,2],[2, -2]]},
                             7: {10207: [[1, 2], [3, -1]],
                                 10107: [[1, 1], [3, -4]]},
                             8: {10208: [[1, 2], [3, -2]],
                                 10108: [[1,1],[4, -4]],
                                 10308: [[1,3],[2, -2]]},
                             9: {10109: [[1,1],[4, -5]],
                                 10209: [[1,2],[4, -1]]},
                             10: {10110: [[1,1],[5, -5]],
                                  10210: [[1,2],[4, -2]],
                                  10410: [[2, -2],[3,2]]},
                             11: {10111: [[1,1],[5, -6]],
                                  10211: [[1,2],[4, -3]]},
                             12: {20206: [[2, 2],[2, -4]],
                                  10412: [[1,4],[3, 0]],
                                  10512: [[2, -2],[3,3]],
                                  10212: [[1,2],[5, -2]],
                                  10112: [[1,1],[5,-7]]},
                             13: {10313: [[1,3],[4, -1]],
                                  10113: [[1, 1],[5, -8]],
                                  10213: [[1,2],[5, -3]]},
                             14: {10114: [[1,1],[5, -9]],
                                  10314: [[1, 3], [4, -2]],
                                  10214: [[1,2],[5, -4]],
                                  10614: [[2, -2], [3, 4]]},
                             15: {10315: [[1,3],[4, -3]],
                                  10415: [[1,4],[4,1]],
                                  10115: [[1,1],[5, -10]],
                                  10215: [[1,2],[5, -5]],
                                  10515: [[1,5],[3, 0]]},
                             16: {20208: [[2,2],[4, -4]],
                                  10616: [[2, -4],[3, 2]],
                                  10716: [[2, -2], [3, 5]],
                                  10116: [[1, 1],[5, -11]],
                                  10316: [[1,3],[5, -1]],
                                  10216: [[1, 2],[5, -6]]},
                             17: {10117: [[1,1],[5, -12]],
                                  10317: [[1,3],[5, -2]],
                                  10217: [[1,2],[5, -7]]},
                             18: {10318: [[1,3],[5, -3]],
                                  10418: [[1,4],[4, -2]],
                                  10218: [[1,2],[5, -8]],
                                  10518: [[3, -3], [4, 2]],
                                  10818: [[2,2],[5, 4]]},
                             19: {10719: [[2, -5],[3, 2]]},
                             20: {10420: [[1, 4], [5, 0]],
                                  10320: [[1,3], [5, -5]],
                                  10520: [[1, 5],[4,0]],
                                  10820: [[2, -4],[3, 4]]},
                             21: {10421: [[1, 4],[5, -1]],
                                  10721: [[1, 7], [3, 0]],
                                  10621: [[3, -3],[4, 3]]},
                             22: {10422: [[1,4],[5, -2]]},
                             23: {10423: [[1, 4], [5, -3]]},
                             24: {10424: [[1,4],[5, -4]],
                                  20212: [[2,2],[6, -6]]},
                             25: {50005: [[5,0],[0, 5]]},
                             27: {30309: [[3,3],[3, -6]]},
                             28: {20414: [[2, 4],[6, -2]]},
                             30: {11130: [[3,3],[5, -5]]},
                             31: {10531: [[1, 5], [5, -6]]},
                             32: {20416: [[2,4],[6, -4]],
                                  40008: [[4, 0],[0,8]]},
                             36: {60006: [[6,0],[0,6]]},
                             39: {11639: [[2, -7],[5, 2]]},
                             42: {10742: [[1,7],[5, -7]]},
                             45: {11945: [[2, -7], [5,5]]},
                             48: {40412: [[4,4],[4, -8]]},
                             49: {70007: [[7,0], [0, 7]]},
                             #196: {1400014: [[14,0],[14,0]]},
                             }

        # either get t1, t2 vectors from simulation torus or from the dicitonary
        try:
            mat = self.sim_tor_dict[self.n][int(self.id)]

            t1_comp = np.array(mat[0]).transpose()
            t2_comp = np.array(mat[1]).transpose()
            # print("dict used instead of utlf matrix for t_1 and t_2 vectors")

            l0 = [int(el) for el in [0., t1_comp[0], t2_comp[0], t1_comp[0] + t2_comp[0]]]
            max_n1 = np.amax(l0)
            min_n1 = np.amin(l0)

            l1 = [int(el) for el in [0., t1_comp[1], t2_comp[1], t1_comp[1] + t2_comp[1]]]
            max_n2 = np.amax(l1)
            min_n2 = np.amin(l1)

            self.t1 = t1_comp[0] * self.a1 + t1_comp[1] * self.a2
            self.t2 = t2_comp[0] * self.a1 + t2_comp[1] * self.a2

        except KeyError:
            self.t1 = self.simulation_torus[:, 0]
            self.t2 = self.simulation_torus[:, 1]

            max_n1 = int(utlf_a)
            max_n2 = int(utlf_b + utlf_d)
            min_n1 = 0
            min_n2 = 0

        c = 0

        t_base_change = np.linalg.inv(np.column_stack([self.t1, self.t2]))

        # base points of the cells accepted so far, i.e. the coordinates without the base vectors
        base_points = []

        # try n1, n2 such that n1*a1 + n2*a2 is in simulation torus
        for n2 in range(min_n2, max_n2):
            for n1 in range(min_n1, max_n1):
                # base point in lattice
                site_vec = n1 * self.a1 + n2 * self.a2

                # skip if an equivalent site is already there. Two sites are equivalent if they
                # differ by any torus vector, i.e. if the difference has integer coordinates in
                # the t basis. Comparing against a fixed list of shifts (+-t1, +-t2, ...) is not
                # enough: the accepted region below is |.| < 1 in both t coordinates, so sites on
                # opposite edges of it differ by 2*t1 or 2*t2 and were kept as separate sites.
                if len(base_points) > 0:
                    shifts = t_base_change @ (site_vec - np.array(base_points)).transpose()
                    if np.any(np.all(np.isclose(shifts, np.round(shifts)), axis=0)):
                        continue

                # stay in positive sector in t basis for both sublattice points
                vec_t_base = t_base_change @ site_vec

                # assert basis change worked
                assert (np.isclose(vec_t_base[0] * self.t1 + vec_t_base[1] * self.t2, site_vec).all())

                if not ((0 <= np.abs(vec_t_base[0]) < 1) and (0 <= np.abs(vec_t_base[1]) < 1)):
                    continue

                for c_b, base_vec in enumerate(self.base_vectors):
                    self.coordinates[c + c_b] = site_vec + base_vec
                c += c_b + 1
                base_points.append(site_vec)

                # if already enough sites found, assume rest are duplicates
                if c > self.n - 1:
                    break

            # the break above only leaves the inner loop, so stop the outer one as well
            if c > self.n - 1:
                break

        if c != self.n:
            print(c, max_n2 * max_n1 * len(self.base_vectors))
            print(self.n, id)
            raise AssertionError("site number not correct")

        # sort coordinates first by x component, then by y component
        # then by going through coordinates we go row by row upwards and in each row from left to right
        ind = np.lexsort((self.coordinates[:, 0], self.coordinates[:, 1]))
        self.coordinates = self.coordinates[ind]

        """get ids for sites"""
        # get y coordinates with the number of nodes which have it
        ys, y_count = np.unique(self.coordinates[:, 1], return_counts=True)
        x_id = 0
        y_id = 0
        i = 0

        while i < self.n:
            # go through each y-coordinate / row in lattice
            for row_el in range(y_count[y_id]):
                # assign id to list
                self.node_ids[i] = [y_id, x_id]

                # raise column index
                x_id += 1

                # raise counter
                i += 1

            # reset column index
            x_id = 0

            # raise row index
            y_id += 1

        # horizontal strings
        all_sites = list(np.arange(0, n))
        while(len(all_sites)>0):
            start_site = all_sites[0]
            neighbor = -1
            this_string = []
            curr_site = start_site
            while neighbor != start_site:
                neighbor = self.neighbor_r(curr_site)
                this_string.append(neighbor)
                try:
                    all_sites.remove(neighbor)
                except ValueError:
                    raise ValueError("Site belongs to several strings of the same type!!")
                curr_site = neighbor
            self.horizontal_strings.append(this_string)

        # up right strings
        all_sites = list(np.arange(0, n))
        while(len(all_sites)>0):
            start_site = all_sites[0]
            neighbor = -1
            this_string = []
            curr_site = start_site
            while neighbor != start_site:
                neighbor = self.neighbor_ur(curr_site)
                this_string.append(neighbor)
                try:
                    all_sites.remove(neighbor)
                except ValueError:
                    raise ValueError("Site belongs to several strings of the same type!!")
                curr_site = neighbor
            self.diagonal_strings_ur.append(this_string)

        # up left strings
        all_sites = list(np.arange(0, n))
        while(len(all_sites)>0):
            start_site = all_sites[0]
            neighbor = -1
            this_string = []
            curr_site = start_site
            while neighbor != start_site:
                neighbor = self.neighbor_ul(curr_site)
                this_string.append(neighbor)
                try:
                    all_sites.remove(neighbor)
                except ValueError:
                    raise ValueError("Site belongs to several strings of the same type!!")
                curr_site = neighbor
            self.diagonal_strings_ul.append(this_string)
            return


    def neighbor_ind_to_ind(self, num):
        """
        6 neighbors of each triagonal lattice point in the order such that Majorana_Hubbard works as desired
        (counter-clockwise starting with bottom left)
        :param num: index of current site
        :return: list of integers that are the neighbors in counter-clockwise order starting with bottom left
        """

        # coordinates of current site
        my_coords = self.coordinates[num]

        # vectors that will be added to take periodic boundary conditions into account
        # Note: 0,0 is also part here to take the trivial case (neighbor in the middle of the lattice) into account
        torus_vec = [np.array([0, 0]).transpose(), self.t1, self.t2, -self.t1, -self.t2, self.t1 + self.t2, -1 * (self.t1 + self.t2),
                     self.t1 - self.t2, self.t2 - self.t1]

        # the four  vectors of neighbor directions
        neighbor_directions = self.base_vectors[1:] + [-a for a in self.base_vectors[1:]]#[(-1)*self.a2, s3@self.a2, self.a1, self.a2, (-1)*s3@self.a2, (-1)*self.a1]

        # list of neighbors
        neighbors = []

        # for each neighbor diretion
        for n_dir in neighbor_directions:

            # coordinate of the neighbor if its not at the boundary
            n_coord = my_coords + n_dir

            # determine if coordinate is a lattice site for the neighbor coordinate plus possible torus vector
            for t_vec in torus_vec:
                try:
                    # Get index of coodinate vector in coordinate array
                    n1 = where_vec_array(self.coordinates, n_coord+t_vec)[0]
                except IndexError:
                    continue
                else:
                    # when successfull append to neighbor list
                    neighbors.append(n1)
                    break
            else:
                #raise AssertionError("Neighbor not found")
                pass

        assert(len(neighbors) == 4)
        return neighbors

    def neighbor_ur(self, ind):
        return self.neighbor_ind_to_ind(ind)[3]

    def neighbor_ul(self, ind):
        return self.neighbor_ind_to_ind(ind)[4]

    def neighbor_r(self, ind):
        return self.neighbor_ind_to_ind(ind)[2]

    def calc_adjacency_matrix(self):
        """
        Calculates the adjacency matrix of the honeycomb graph
        :return:
        """
        for i in range(self.n):
            for j in range(self.n):
                if j in self.neighbor_ind_to_ind(i):
                    self.adjacencymatrix[i][j] = 1
                else:
                    self.adjacencymatrix[i][j] = 0

    def distance(self, i, j):
        ci = self.coordinates[i]
        cj = self.coordinates[j]
        return np.linalg.norm(ci - cj)

    def pacific_neighbors(self, i, j):
        """
        Given two Site i,j (integers) that are neighbors, return True if their connection is "on the other side",
        i.e. due to periodic boundary conditions
        :param i: integer in 0... n-1
        :param j: integer in 0... n-1
        :return: bool
        """
        if self.distance(i, j) > self.a * 1.005:
            return True
        else:
            return False

    def simple_cycles(self, i):
        if self.adjacencymatrix[0][0] == -1:
            self.calc_adjacency_matrix()

        # create graph
        G = nx.Graph()

        # go through all sites
        for i in range(self.n):
            maj_label = lambda v: int(np.where(self.new_order == v)[0])

            # add nodes
            G.add_node(tuple(self.node_ids[i]), pos=tuple(self.coordinates[i]), node_label= maj_label(i))

            # add edges
            for j in range(i + 1, self.n):
                if self.adjacencymatrix[i][j] == 1:
                    if self.pacific_neighbors(i, j):
                        G.add_edge(tuple(self.node_ids[i]), tuple(self.node_ids[j]))
            return nx.all_simple_paths(G, source=i, target=i)


    def plot(self, save_fig=False, highlight = None, highlight_triangles = None):
        """
        Creates and draws networkx graph
        :return:
        """
        # if not calculated so far, calc adjacency matrix
        if self.adjacencymatrix[0][0] == -1:
            self.calc_adjacency_matrix()

        # create graph
        G = nx.Graph()
        color = "dodgerblue"

        fig = plt.figure()
        fig.set_tight_layout(False)

        # go through all sites
        for i in range(self.n):
            maj_label = lambda v: int(np.where(self.new_order == v)[0])

            # add nodes
            G.add_node(tuple(self.node_ids[i]), pos=tuple(self.coordinates[i]), col=color, node_label= maj_label(i))

            # add edges
            for j in range(i + 1, self.n):
                if self.adjacencymatrix[i][j] == 1:
                    if self.pacific_neighbors(i, j):
                        st = "dotted"
                    else:
                        st = "solid"
                    # vertical link?
                    if self.coordinates[i][0] == self.coordinates[j][0]:
                        edge_label = int(maj_label(i)/2)
                    else:
                        edge_label = ""
                        pass

                    G.add_edge(tuple(self.node_ids[i]), tuple(self.node_ids[j]), style=st, label=edge_label)

        pos = nx.get_node_attributes(G, 'pos')

        colors = nx.get_node_attributes(G, 'col')
        color_list = [colors[idd] for idd in G.nodes]
        if highlight is not None:
            for el in highlight:
                color_list[el] = 'orange'
        node_labels = nx.get_node_attributes(G, 'node_label')
        # index_list = [indices[idd] for idd in G.nodes]

        styles = nx.get_edge_attributes(G, 'style')
        style_list = [styles[idd] for idd in G.edges]

        labels = nx.get_edge_attributes(G, 'label')
        label_list = [labels[idd] for idd in G.edges]
        print(labels)
        nx.draw_networkx_edges(G, pos, style=style_list)
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels, rotate=False)
        nx.draw_networkx_nodes(G, pos, node_color=color_list)

        # Choose here which labels to draw
        nx.draw_networkx_labels(G, pos, node_labels)

        plt.arrow(0, 0, self.t1[0], self.t1[1], width=0.01, head_width=0.15, head_length=0.2, color="green", alpha=1.0)
        plt.arrow(0, 0, self.t2[0], self.t2[1], width=0.01, head_width=0.15, head_length=0.2, color="green", alpha=1.0)
        #custom_lines = [Line2D([0], [0], color="white", marker='o', markerfacecolor="dodgerblue"),
        #                Line2D([0], [0], color="white", marker='o', markerfacecolor="orange")]
        #plt.legend(custom_lines, ["Sublattice A", "Sublattice B"])
        if highlight_triangles is not None:
            for tri in highlight_triangles:
                xs = [self.coordinates[i][0] for i in tri]
                ys = [self.coordinates[i][1] for i in tri]
                plt.fill(xs, ys, 'orange')
        if save_fig:
            plt.savefig("Lattices/Triangular/{}/Triangular{}.png".format(self.n, self.id))
        plt.show()
        return

    def plot_extended(self, num_x, num_y, save_fig=False, save_name=None,
                      highlight_sites = None, highlight_sites_colors=None,
                      highlight_triangles = None, highlight_triangles_colors = None,
                      lowlight_triangles=None, intersection_color='red',
                      legend=False, labels=None, title=None,
                      dont_show=False, img_path=None):

        """
                Creates and draws networkx graph
                :return:
                """
        # if not calculated so far, calc adjacency matrix
        if self.adjacencymatrix[0][0] == -1:
            self.calc_adjacency_matrix()

        # create graph
        G = nx.Graph()
        color = "dodgerblue"

        fig = plt.figure()
        fig.set_tight_layout(False)

        # go through all sites
        for i in range(self.n):
            for i_x in range(num_x):
                for i_y in range(num_y):

                    # add nodes
                    p = self.coordinates[i] + i_x*self.t1 + i_y*self.t2
                    G.add_node(tuple([i, i_x, i_y]), pos=tuple(p), col=color, node_label=i)

        pos = nx.get_node_attributes(G, 'pos')

        for c1, (k1, v1) in enumerate(pos.items()):
            for c2, (k2, v2) in enumerate(pos.items()):
                if c1<=c2:
                    continue

                if np.linalg.norm(np.array(v1) - np.array(v2)) < self.a + 0.001:
                    G.add_edge(k1, k2, style="solid")

        colors = nx.get_node_attributes(G, 'col')
        color_list = [colors[idd] for idd in G.nodes]

        node_labels = nx.get_node_attributes(G, 'node_label')
        # index_list = [indices[idd] for idd in G.nodes]

        styles = nx.get_edge_attributes(G, 'style')
        style_list = [styles[idd] for idd in G.edges]

        #labels = nx.get_edge_attributes(G, 'label')
        label_list = ["" for idd in G.edges]
        nx.draw_networkx_edges(G, pos)
        #nx.draw_networkx_edge_labels(G, pos=pos,  rotate=False, label_list)

        if highlight_sites is not None:

            assert(len(highlight_sites) == len(highlight_sites_colors))
            assert('red' not in highlight_sites_colors)

            for highlight_set, set_color in zip(highlight_sites, highlight_sites_colors):
                for h_site in highlight_set:
                    for i_x in range(num_x):
                        for i_y in range(num_y):
                            ind = list(G.nodes).index((h_site, i_x, i_y))
                            color_list[ind] = set_color
            try:
                intersection = list(np.intersect1d(*highlight_sites))
                if len(intersection) > 0:
                    for h_site in intersection:
                        for i_x in range(num_x):
                            for i_y in range(num_y):
                                ind = list(G.nodes).index((h_site, i_x, i_y))
                                color_list[ind] = 'red'
            except TypeError:
                pass


        nx.draw_networkx_nodes(G, pos, node_color=color_list)

        # Choose here which labels to draw
        nx.draw_networkx_labels(G, pos, node_labels)

        plt.arrow(self.coordinates[0,0], self.coordinates[0,1], self.t1[0], self.t1[1], width=0.01, head_width=0.15, head_length=0.2, color="green", alpha=1.0)
        plt.arrow(self.coordinates[0,0], self.coordinates[0,1], self.t2[0], self.t2[1], width=0.01, head_width=0.15, head_length=0.2, color="green", alpha=1.0)

        if highlight_triangles is not None:
            for h_tri_set, h_tri_col in zip(highlight_triangles, highlight_triangles_colors):

                # for triangle in highlight triangle set
                for tri in h_tri_set:

                    tri_nodes = []
                    for c_tri, tri_site in enumerate(tri):

                        # list for each site
                        tri_nodes.append([])

                        for node_tuple in G.nodes:
                            if node_tuple[0] == tri_site:
                                tri_nodes[c_tri].append(node_tuple)

                    for site1 in tri_nodes[0]:
                        for site2 in tri_nodes[1]:
                            for site3 in tri_nodes[2]:
                                p1 = np.array(pos[site1])
                                p2 = np.array(pos[site2])
                                p3 = np.array(pos[site3])
                                if np.linalg.norm(p1 - p2) < self.a + 0.001:
                                    if np.linalg.norm(p3 - p2) < self.a + 0.001:
                                        if np.linalg.norm(p1-p3) < self.a + 0.001:
                                            # assert only downpointing!
                                            if p1[1] < p2[1] and p1[1] < p3[1]:
                                                xs = [p1[0], p2[0], p3[0]]
                                                ys = [p1[1], p2[1], p3[1]]

                                                plt.fill(xs, ys, h_tri_col)

            # intersections = same triangles
            if len(highlight_triangles) == 2 and isinstance(highlight_triangles[0][0], list):
                intersections = []

                for tri in highlight_triangles[0]:
                    if tri in highlight_triangles[1]:
                        intersections.append(tri)

                for tri in intersections:
                    tri_nodes = []
                    for c_tri, tri_site in enumerate(tri):

                        # list for each site
                        tri_nodes.append([])

                        for node_tuple in G.nodes:
                            if node_tuple[0] == tri_site:
                                tri_nodes[c_tri].append(node_tuple)

                    for site1 in tri_nodes[0]:
                        for site2 in tri_nodes[1]:
                            for site3 in tri_nodes[2]:
                                p1 = np.array(pos[site1])
                                p2 = np.array(pos[site2])
                                p3 = np.array(pos[site3])
                                if np.linalg.norm(p1 - p2) < self.a + 0.001:
                                    if np.linalg.norm(p3 - p2) < self.a + 0.001:
                                        if np.linalg.norm(p1-p3) < self.a + 0.001:
                                            if p1[1] < p2[1] and p1[1] < p3[1]:
                                                xs = [p1[0], p2[0], p3[0]]
                                                ys = [p1[1], p2[1], p3[1]]
                                                plt.fill(xs, ys, intersection_color)

            elif len(highlight_triangles) > 2:
                raise ValueError('Cannot deal with three set of highlight triangles so far')


        if lowlight_triangles is not None:
            for tri in lowlight_triangles:
                tri_nodes = []
                for c_tri, tri_site in enumerate(tri):

                    # list for each site
                    tri_nodes.append([])

                    for node_tuple in G.nodes:
                        if node_tuple[0] == tri_site:
                            tri_nodes[c_tri].append(node_tuple)

                for site1 in tri_nodes[0]:
                    for site2 in tri_nodes[1]:
                        for site3 in tri_nodes[2]:
                            p1 = np.array(pos[site1])
                            p2 = np.array(pos[site2])
                            p3 = np.array(pos[site3])
                            if np.linalg.norm(p1 - p2) < self.a + 0.001:
                                if np.linalg.norm(p3 - p2) < self.a + 0.001:
                                    if np.linalg.norm(p1-p3) < self.a + 0.001:
                                        if p1[1] < p2[1] and p1[1] < p3[1]:
                                            xs = [p1[0], p2[0], p3[0]]
                                            ys = [p1[1], p2[1], p3[1]]
                                            plt.fill(xs, ys, 'silver')
        if title is not None:
            plt.title(title)
        if legend:
            bg_col = mpl.rcParams["axes.facecolor"]
            custom_lines = [Line2D([0], [0], marker='v', color=bg_col, markerfacecolor=c, markersize=15) for c in highlight_triangles_colors]
            custom_lines.append(Line2D([0], [0], marker='v', color=bg_col, markerfacecolor=intersection_color, markersize=15))
            custom_lines.append(Line2D([0], [0], marker='o', color=bg_col , label='Scatter',
                                       markerfacecolor=highlight_sites_colors[0], markersize=15))
            plt.legend(custom_lines, labels, loc=2, fontsize='xx-large')

        if save_fig:
            if save_name is None:
                raise ValueError
            save_name = f"n{self.n}_ID{self.id}_{save_name}"
            # savefig(save_name, meta_description="", timestamp=False, destination=img_path)
            plt.savefig(save_name, format="pdf")
        if not dont_show:
            plt.show()
        else:
            plt.close()
        return



    def torus_coordinates(self, x, y, R=6, r=4, in_tbasis=False):
        '''
        in_tbasis: False:
            Given any (x,y) coordinates of a point in the plot, return x,y,z the cartesian coordinates of the corresponding 3D torus points
        in_tbasis: True:
        Given x,y in [0,1) the components of any vector in the lattice in simulation torus basis
            return x,y,z the cartesian coordinates of the corresponding 3D torus point
        :param x:
        :param y:
        :param R:
        :param r:
        :param in_tbasis:
        :return:
        '''

        if not in_tbasis:
            # components in terms of torus simulation vectors
            inv = np.linalg.inv(np.column_stack([self.t1, self.t2]))
            cx = inv[0,0] * x + inv[0,1] * y
            cy = inv[1,0] * x + inv[1,1] * y
        else:
            cx = x
            cy = y

        # angles
        phi = 2* np.pi*cx
        theta = 2* np.pi * cy

        # cartesian coordinates on the torus surface
        x = (R + r * np.cos(theta)) * np.cos(phi)
        y = (R + r * np.cos(theta)) * np.sin(phi)
        z = (r * np.sin(theta))
        return x, y, z

    def plot_on_torus(self, save_fig=False, lowlight_triangles=None):
        """
            Creates and draws networkx graph
            :return:
            """
        # if not calculated so far, calc adjacency matrix
        if self.adjacencymatrix[0][0] == -1:
            self.calc_adjacency_matrix()

        # create graph
        G = nx.Graph()

        fig = plt.figure()
        ax = Axes3D(fig)
        fig.set_tight_layout(False)

        # go through all sites
        for i in range(self.n):


            # add nodes
            coordinates_torus = self.torus_coordinates(*self.coordinates[i])
            G.add_node(i, pos=tuple(coordinates_torus), col="dodgerblue", node_label=i)

            # add edges
            for j in range(i + 1, self.n):
                if self.adjacencymatrix[i][j] == 1:
                    if self.pacific_neighbors(i, j):
                        st = "dotted"
                    else:
                        st = "solid"

                    G.add_edge(i, j, style=st, label="")

        pos = nx.get_node_attributes(G, 'pos')

        colors = nx.get_node_attributes(G, 'col')
        color_list = [colors[idd] for idd in G.nodes]

        node_labels = nx.get_node_attributes(G, 'node_label')
        # index_list = [indices[idd] for idd in G.nodes]

        styles = nx.get_edge_attributes(G, 'style')
        style_list = [styles[idd] for idd in G.edges]

        labels = nx.get_edge_attributes(G, 'label')
        label_list = [labels[idd] for idd in G.edges]

        #nx.draw_networkx_edges(G, pos, style=style_list)
        #nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels, rotate=False)
        #nx.draw_networkx_nodes(G, pos, node_color=color_list)

        # Choose here which labels to draw
        #nx.draw_networkx_labels(G, pos, node_labels)

        #plt.arrow(0, 0, self.t1[0], self.t1[1], width=0.01, head_width=0.15, head_length=0.2, color="green", alpha=1.0)
        #plt.arrow(0, 0, self.t2[0], self.t2[1], width=0.01, head_width=0.15, head_length=0.2, color="green", alpha=1.0)

        # plot the torus itself:
        xs = np.linspace(0, 1, 50)
        ys = np.linspace(0, 1, 50)
        X, Y = np.meshgrid(xs, ys)

        ax.plot_surface(*self.torus_coordinates(X, Y, in_tbasis=True), alpha=0.4, color='blue')

        for key, values in pos.items():
            ax.scatter(values[0], values[1], values[2], color='red', s=20)
            ax.text(*values, key)

        for i,j, edge_props in G.edges(data=True):
            if edge_props['style'] == "solid":
                xs = np.linspace(self.coordinates[i][0], self.coordinates[j][0], 25)
                ys = np.linspace(self.coordinates[i][1], self.coordinates[j][1], 25)
                ax.plot(*self.torus_coordinates(xs, ys), color="black")
            else:
                i_xy = np.array([self.coordinates[i][0], self.coordinates[i][1]])
                j_xy = np.array([self.coordinates[j][0], self.coordinates[j][1]])

                for t_vec in [self.t1, self.t2, -self.t1, -self.t2, self.t1 + self.t2, -1 * (self.t1 + self.t2),
                              self.t1 - self.t2, self.t2 - self.t1]:
                    if np.linalg.norm((i_xy+t_vec)-j_xy) < self.a + 0.01:
                        inv = np.linalg.inv(np.column_stack([self.t1, self.t2]))
                        i_cx = inv[0, 0] * self.coordinates[i][0] + inv[0, 1] * self.coordinates[i][1]
                        i_cy = inv[1, 0] * self.coordinates[i][0] + inv[1, 1] * self.coordinates[i][1]
                        j_cx = inv[0, 0] * self.coordinates[j][0] + inv[0, 1] * self.coordinates[j][1]
                        j_cy = inv[1, 0] * self.coordinates[j][0] + inv[1, 1] * self.coordinates[j][1]

                        if np.allclose(t_vec, self.t1):
                            t1_c = np.linspace(i_cx, j_cx+1)%1
                            t2_c = np.linspace(i_cy, j_cy)
                        elif np.allclose(t_vec, -self.t1):
                            t1_c = np.linspace(i_cx+1, j_cx)%1
                            t2_c = np.linspace(i_cy, j_cy)
                        elif np.allclose(t_vec, self.t2):
                            t1_c = np.linspace(i_cx, j_cx)
                            t2_c = np.linspace(i_cy, j_cy+1)%1
                        elif np.allclose(t_vec, -self.t2):
                            t1_c = np.linspace(i_cx, j_cx)
                            t2_c = np.linspace(i_cy+1, j_cy)%1
                        elif np.allclose(t_vec, self.t1 + self.t2):
                            t1_c = np.linspace(i_cx, j_cx+1)%1
                            t2_c = np.linspace(i_cy, j_cy+1)%1
                        elif np.allclose(t_vec, -self.t1 + self.t2):
                            t1_c = np.linspace(i_cx+1, j_cx)%1
                            t2_c = np.linspace(i_cy, j_cy+1)%1
                        elif np.allclose(t_vec, -self.t1 - self.t2):
                            t1_c = np.linspace(i_cx+1, j_cx)%1
                            t2_c = np.linspace(i_cy+1, j_cy)%1
                        elif np.allclose(t_vec, self.t1 - self.t2):
                            t1_c = np.linspace(i_cx, j_cx+1)%1
                            t2_c = np.linspace(i_cy+1, j_cy)%1
                        else:
                            raise ValueError

                        ax.plot(*self.torus_coordinates(t1_c, t2_c, in_tbasis=True), color='black', linestyle='dashed')

        if save_fig:
            plt.savefig("Lattices/Honeycomb/{}/Honeycomb{}_torus.png".format(self.n, self.id))
        plt.show()
        return




    def draw_basic_network(self, fig, ax):
        """
        Creates and draws networkx graph
        :param highlight: indices 0...n that will be plotted in another color
        :return:
        """
        # if not calculated so far, calc adjacency matrix
        if self.adjacencymatrix[0][0] == -1:
            self.calc_adjacency_matrix()

        # create graph
        G = x.Graph()

        # fig = plt.figure()
        # fig.set_tight_layout(False)

        # go through all sites
        for i in range(self.n):

            color = "dodgerblue"

            # add nodes
            G.add_node(tuple(self.node_ids[i]), pos=tuple(self.coordinates[i]), col=color)

            # add edges
            for j in range(i + 1, self.n):
                if self.adjacencymatrix[i][j] == 1:
                    if self.pacific_neighbors(i, j):
                        st = "dotted"
                    else:
                        st = "solid"
                    G.add_edge(tuple(self.node_ids[i]), tuple(self.node_ids[j]), style=st)

        pos = nx.get_node_attributes(G, 'pos')
        colors = nx.get_node_attributes(G, 'col')
        color_list = [colors[idd] for idd in G.nodes]
        styles = nx.get_edge_attributes(G, 'style')
        style_list = [styles[idd] for idd in G.edges]
        nx.draw(G, pos, ax=ax, node_color=color_list, style=style_list, with_labels=False)
        ax.arrow(0, 0, self.t1[0], self.t1[1], width=0.01, head_width=0.15, head_length=0.2, color="green", alpha=1.0)
        ax.arrow(0, 0, self.t2[0], self.t2[1], width=0.01, head_width=0.15, head_length=0.2, color="green", alpha=1.0)
        return

    def draw_highlights(self, highlights, fig, ax, lowlight_color="dodgerblue", highlight_color="orange", with_labels=True):
        new_axis = fig.add_axes(ax)
        G = nx.Graph()
        for i in range(self.n):

            # # if site in highlight list
            if i in highlights:
                color = highlight_color
                if with_labels:
                    label = highlights.index(i)
                else:
                    label = ""
            else:
                color = lowlight_color
                label = " "

            # add nodes
            G.add_node(tuple(self.node_ids[i]), pos=tuple(self.coordinates[i]), col=color, label=label)

        pos = nx.get_node_attributes(G, 'pos')
        colors = nx.get_node_attributes(G, 'col')
        labels = nx.get_node_attributes(G, "label")
        color_list = [colors[idd] for idd in G.nodes]
        nx.draw(G, pos, ax=new_axis, node_color=color_list, labels=labels)
        return new_axis


def create_lattice_dict(n_latt, which="Honeycomb", save_fig=False):
    """
    For a given lattice list (n, lattice id),(n2, lattice id2), ... and a chosen lattice type
    create a lattice dictionary and store it in data/latt_dict_triangular.pkl or ..._honeycomb.pkl
    :param n_latt: list of lists [[n1, lattice id 1], [n2, lattice id2], [n3, lattice id3]]
    :param which: string either "honeycomb" or "triangular" not case sensitive
    :return:
    """
    latt_dict = {}

    # For every cluster
    for (n, latt_id) in n_latt:
        if n not in latt_dict:
            latt_dict[n] = {}
        latt_dict[n][latt_id] = {}

        # Honeycomb
        if which.lower() == "honeycomb":
            x = Honeycomb(n, latt_id, order="paper")
            if save_fig:
                x.plot(save_fig=True)
            latt_dict[n][latt_id]["horizontal strings"] = x.horizontal_strings
            latt_dict[n][latt_id]["vertical strings"] = x.vertical_strings
            latt_dict[n][latt_id]["new_order"] = x.new_order
            latt_dict[n][latt_id]["b"] = x.ind_of_sublattice(0)
            latt_dict[n][latt_id]["a"] = x.ind_of_sublattice(1)

        # Triangular
        elif which.lower() == "triangular":
            x = Triangular(n, latt_id)
            latt_dict[n][latt_id]["horizontal strings"] = x.horizontal_strings
            latt_dict[n][latt_id]["diagonal up-right strings"] = x.diagonal_strings_ur
            latt_dict[n][latt_id]["diagonal up-left strings"] = x.diagonal_strings_ul
            if save_fig:
                x.plot(save_fig=True)

        else:
            raise TypeError("Type of Lattice not known")

        for i in range(n):
            latt_dict[n][latt_id][i] = x.neighbor_ind_to_ind(i)
    p = os.path.join(os.getcwd(), "resources/latticedicts")
    if not os.path.exists(p):
        os.makedirs(p)
    with open(p+"/latt_dict_{}.pkl".format(which), "wb") as f:
        pickle.dump(latt_dict, f)
    print("Lattice Dict saved in resources/latt_dict_{}.pkl".format(which))
    return


def add_to_lattice_dict(n_latt, which="Honeycomb"):
    """
    given a list of lists (number of sites, lattice id) append it to the lattice dictionary
    :param n_latt:
    :return:
    """

    assert 0, "WARNING this function is MAYBE NOT UP TO DATE "

    # open dict file
    with open("data/latt_dict_{}.pkl".format(which), "rb") as handle:
        latt_dict = pickle.load(handle)

    for (n, latt_id) in n_latt:
        if n not in latt_dict:
            latt_dict[n] = {}
        if latt_id in latt_dict[n]:
            continue

        latt_dict[n][latt_id] = {}
        if which.lower()=="honeycomb":
            x = Honeycomb(n, latt_id, order="paper")
            latt_dict[n][latt_id]["horizontal strings"] = x.horizontal_strings
            latt_dict[n][latt_id]["vertical strings"] = x.vertical_strings
            latt_dict[n][latt_id]["new_order"] = x.new_order
            latt_dict[n][latt_id]["b"] = x.ind_of_sublattice(0)
            latt_dict[n][latt_id]["a"] = x.ind_of_sublattice(1)

        elif which.lower() =="triangular":
            x = Triangular(n, latt_id)

            latt_dict[n][latt_id]["horizontal strings"] = x.horizontal_strings
            latt_dict[n][latt_id]["diagonal up-right strings"] = x.diagonal_strings_ur
            latt_dict[n][latt_id]["diagonal up-left strings"] = x.diagonal_strings_ul

        else:
            raise ValueError("Type of Lattice not known")

        for i in range(n):
            latt_dict[n][latt_id][i] = x.neighbor_ind_to_ind(i)
    f = open("data/latt_dict_{}.pkl".format(which), "wb")
    pickle.dump(latt_dict, f)
    f.close()





if __name__ == '__main__':
    create_lattice_dict(n_latt_triangular_very_large, which='triangular')
    #x = Triangular(9, 30003)
    #print(x.diagonal_strings_ul)