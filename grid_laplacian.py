from collections import defaultdict
from functools import reduce

from IPython.core.debugger import Tracer  # noqa
import graph_tool as gt
from graph_tool.spectral import laplacian
import matplotlib  # noqa
import matplotlib.cm as cm  # noqa
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from numpy import arctan, cos, pi, sin
import plotly.graph_objs
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot  # noqa
import plotly.plotly as py  # noqa
import scipy.linalg

from spherical_voronoi import SphericalVoronoi


# TODO
#   think about using symmetry of icosahedron to copy some attributes, like
#       edge length, in the refinement method

class Tetrahedron(object):
    """ class to hold the vertices, edges and faces of a regular tetrahedron
    """
    def __init__(self):
        self.vertices = [
                (1, 1, 1),
                (1, -1, -1),
                (-1, 1, -1),
                (-1, -1, 1)
                ]
        self.vertices = np.array(self.vertices, dtype='float')
        self.vertices = self.vertices * (1/np.sqrt(3))
        self.edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        self.faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]


class Isocahedron(object):
    """ class to hold the vertices, edges and faces of a regular isocahedron
    # https://en.wikipedia.org/wiki/Regular_icosahedron
    """
    def __init__(self):
        h1 = sin(arctan(1/2))  # height of the five vertices below the top
        h2 = sin(arctan(-1/2))  # height of the five vertices above the bottom
        d = cos(arctan(1/2))  # planar radius (xy-plane) of the middle vertices
        self.vertices = [
            (0, 0, 1),  # 0
            (d*cos(0), d*sin(0), h1),  # 1
            (d*cos(2*pi/5), d*sin(2*pi/5), h1),  # 2
            (d*cos(4*pi/5), d*sin(4*pi/5), h1),  # 3
            (d*cos(6*pi/5), d*sin(6*pi/5), h1),  # 4
            (d*cos(8*pi/5), d*sin(8*pi/5), h1),  # 5
            (d*cos(1*pi/5), d*sin(1*pi/5), h2),  # 6
            (d*cos(3*pi/5), d*sin(3*pi/5), h2),  # 7
            (d*cos(5*pi/5), d*sin(5*pi/5), h2),  # 8
            (d*cos(7*pi/5), d*sin(7*pi/5), h2),  # 9
            (d*cos(9*pi/5), d*sin(9*pi/5), h2),  # 10
            (0, 0, -1)  # 11
            ]
        self.vertices = np.array(self.vertices)

        self.faces = [
            (0, 1, 2),
            (0, 2, 3),
            (0, 3, 4),
            (0, 4, 5),
            (0, 5, 1),
            (1, 2, 6),
            (2, 3, 7),
            (3, 4, 8),
            (4, 5, 9),
            (5, 1, 10),
            (7, 6, 2),
            (8, 7, 3),
            (9, 8, 4),
            (10, 9, 5),
            (6, 10, 1),
            (11, 6, 7),
            (11, 7, 8),
            (11, 8, 9),
            (11, 9, 10),
            (11, 10, 6)
        ]

        self.edges = [
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 1),
            (1, 6),
            (6, 2),
            (2, 7),
            (7, 3),
            (3, 8),
            (8, 4),
            (4, 9),
            (9, 5),
            (5, 10),
            (10, 1),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (10, 6),
            (11, 6),
            (11, 7),
            (11, 8),
            (11, 9),
            (11, 10),
        ]


class Grid(object):
    def __init__(self, vertices, edges, faces):
        """ base class

        Parameters
        ----------
        vertices : list of d-tuples or np.array of shape (n, d)
            corresponding to the coordinates of the graph in d dimensions
        edges : list of 2-tuples (v_in, v_out) corresponding to the
            vertex indices connected by the edges
        faces : list or set of tuples (v_0, v_1, v_2, ...) corresponding
            to the vertex indices restricting the faces
        """

        self.graph = gt.Graph(directed=False)
        self.graph.add_vertex(len(vertices))
        self.graph.add_edge_list(edges)
        assert all([len(f) > 2 for f in faces])
        self.faces = faces
        self.vertices = vertices
        self.dimension = self.vertices.shape[1]

    def get_barycentric_dual(self):
        """ calculate the dual grid, where the dual vertices are the
            barycenters of the primal faces
            https://en.wikipedia.org/wiki/Centroid#Centroid_of_polygon

            we use the two-dimensional formula for the xy and the xz-projection
        """
        dual_grid = Grid(np.zeros((0, self.dimension)), [], [])
        # dict storing which primal edges neighbor which dual vertices
        primal_edges_dual_vertices = defaultdict(list)
        # dict storing which primal vertices neighbor which dual vertices
        primal_vertices_dual_vertices = defaultdict(list)
        # iterate over primal faces, getting dual vertices
        for f in self.faces:
            x = self.vertices[f][:, 0]
            y = self.vertices[f][:, 1]
            z = self.vertices[f][:, 2]
            A_xy = 0.5*sum(
                    [(x[i]*y[(i+1) % len(f)] - y[i]*x[(i+1) % len(f)])
                        for i in range(len(f))])
            A_xz = 0.5*sum(
                    [(x[i]*z[(i+1) % len(f)] - z[i]*x[(i+1) % len(f)])
                        for i in range(len(f))])
            # in case to vertices share the same x, y or z-coordinate, we have
            #    to use the yz-projection
            if (np.isclose(A_xy, 0)) or (np.isclose(A_xz, 0)):
                A_yz = 0.5*sum(
                        [(y[i]*z[(i+1) % len(f)] - z[i]*y[(i+1) % len(f)])
                            for i in range(len(f))])
            if not np.isclose(A_xy, 0):
                c_x = sum(
                        [(x[i] + x[(i+1) % len(f)]) *
                            (x[i]*y[(i+1) % len(f)] - y[i]*x[(i+1) % len(f)])
                            for i in range(len(f))])/(6*A_xy)
            else:
                c_x = sum(
                        [(x[i] + x[(i+1) % len(f)]) *
                            (x[i]*z[(i+1) % len(f)] - z[i]*x[(i+1) % len(f)])
                            for i in range(len(f))])/(6*A_xz)
            if not np.isclose(A_xy, 0):
                c_y = sum(
                        [(y[i] + y[(i+1) % len(f)]) *
                            (x[i]*y[(i+1) % len(f)] - y[i]*x[(i+1) % len(f)])
                            for i in range(len(f))])/(6*A_xy)
            else:
                c_y = sum(
                        [(y[i] + y[(i+1) % len(f)]) *
                            (y[i]*z[(i+1) % len(f)] - z[i]*y[(i+1) % len(f)])
                            for i in range(len(f))])/(6*A_yz)

            if not np.isclose(A_xz, 0):
                c_z = sum(
                        [(z[i] + z[(i+1) % len(f)]) *
                            (x[i]*z[(i+1) % len(f)] - z[i]*x[(i+1) % len(f)])
                            for i in range(len(f))])/(6*A_xz)
            else:
                c_z = sum(
                        [(z[i] + z[(i+1) % len(f)]) *
                            (y[i]*z[(i+1) % len(f)] - z[i]*y[(i+1) % len(f)])
                            for i in range(len(f))])/(6*A_yz)

            dual_grid.graph.add_vertex()
            c_index = len(dual_grid.vertices)
            dual_grid.vertices = np.vstack(
                    [dual_grid.vertices, [c_x, c_y, c_z]])
            for i in range(len(f)):
                primal_edges_dual_vertices[
                        tuple(sorted([f[i], f[(i+1) % len(f)]]))
                    ].append(c_index)
                primal_vertices_dual_vertices[f[i]].append(c_index)

        # iterate over primal edges to get the dual edges
        primal_edge_length = dual_grid.graph.new_edge_property('float')
        self.dual_edge_length = self.graph.new_edge_property('float')
        for primal_edge, dual_vertices in (
                primal_edges_dual_vertices.items()):
            if len(dual_vertices) != 2:
                raise Exception()
            v_0, v_1 = primal_edge
            w_0, w_1 = dual_vertices
            e = self.graph.edge(v_0, v_1)
            f = dual_grid.graph.add_edge(w_0, w_1)
            self.dual_edge_length[e] = np.linalg.norm(
                    dual_grid.vertices[w_0] - dual_grid.vertices[w_1])
            primal_edge_length[e] = np.linalg.norm(
                    self.vertices[v_0] - self.vertices[v_1])

        # iterate over primal vertices to get the dual faces, store the primal
        #   vertex - dual face pairing as class attribute
        self.primal_vertex_dual_face = dict()
        for primal_vertex, dual_vertices in (
                primal_vertices_dual_vertices.items()):
            if len(dual_vertices) == 3:
                dual_face = dual_vertices
            else:
                dual_face = []
                v_0 = dual_vertices[i]
                dual_vertices.remove(v_0)
                dual_face.append(v_0)
                for i in range(len(dual_vertices)):
                    for v_1 in dual_vertices:
                        if dual_grid.graph.edge(v_0, v_1):
                            dual_vertices.remove(v_1)
                            dual_face.append(v_1)
                            v_0 = v_1
                            break
                    else:
                        raise Exception()
            dual_face_index = len(dual_grid.faces)
            self.primal_vertex_dual_face.update(
                    {primal_vertex: dual_face_index})
            dual_grid.faces.append(dual_face)

        self.dual_grid = dual_grid
        dual_grid.dual_grid = self
        return dual_grid

    def get_triangulation(self):
        triangles = []
        for f in self.faces:
            for i in range(1, len(f)-1):
                triangles.append([f[0], f[i], f[i+1]])

        self.triangles = triangles
        return triangles


class TriangularGrid(Grid):
    def __init__(self, vertices, edges, faces, refinement=0):
        super().__init__(vertices, edges, faces)

        # use numpy array instad of set arithmetic here
        self.faces = np.array([f for f in self.faces], dtype='int')

        if refinement > 0:
            for i in range(refinement):
                self.refine_mesh()

    def refine_mesh(self):
        """
        refine a triangular grid via intersecting every edge at its midpoint
            and projecting the intersecting vertex on the sphere. This will
            replace the old edges and faces with the new edges and faces
        """
        new_edges = set()
        new_faces = np.zeros((0, self.faces.shape[1]), dtype='int')
        edge_intersecting_vertex = dict()

        for e in list(self.graph.edges()):
            v_0 = self.graph.vertex_index[e.source()]
            v_1 = self.graph.vertex_index[e.target()]
            v_0_coords = self.vertices[v_0]
            v_1_coords = self.vertices[v_1]
            intersecting_vertex_coords = (1/2)*(v_0_coords + v_1_coords)
            intersecting_vertex_coords /= np.linalg.norm(
                    intersecting_vertex_coords)
            intersecting_vertex_index = len(self.vertices)
            self.vertices = np.vstack(
                    [self.vertices, intersecting_vertex_coords])
            self.graph.add_vertex()
            new_edges.add((v_0, intersecting_vertex_index))
            new_edges.add((v_1, intersecting_vertex_index))
            edge_index = self.graph.edge_index[e]
            edge_intersecting_vertex.update(
                    {edge_index: intersecting_vertex_index})

        for f in self.faces:
            """
            #          v_0
            #         /   \
            #       v_01---v_02
            #       /  \  / \
            #    v_1---v_12---v_2
            """
            v_0, v_1, v_2 = f

            e_01 = self.graph.edge_index[v_0, v_1]
            e_02 = self.graph.edge_index[v_0, v_2]
            e_12 = self.graph.edge_index[v_1, v_2]
            v_01 = edge_intersecting_vertex[e_01]
            v_02 = edge_intersecting_vertex[e_02]
            v_12 = edge_intersecting_vertex[e_12]

            new_edges.add((v_01, v_02))
            new_edges.add((v_01, v_12))
            new_edges.add((v_02, v_12))

            new_faces = np.vstack([new_faces, [v_0, v_01, v_02]])
            new_faces = np.vstack([new_faces, [v_1, v_12, v_01]])
            new_faces = np.vstack([new_faces, [v_2, v_02, v_12]])
            new_faces = np.vstack([new_faces, [v_01, v_12, v_02]])

        self.graph.clear_edges()
        self.graph.add_edge_list(list(new_edges))
        self.faces = new_faces


class TriangularGridv2(object):
    def __init__(self, vertices, triangles, refinement=0):
        """ class for a triangular grid in a simple numpy-array based way
        the graph-theoretical approach is omitted here, vertices are stored
        redundant for each triangle and there is no connectivity information
        besides the triangles.

        Parameters
        ----------
        vertices : list of d-tuples or np.array of shape (n, d)
            corresponding to the coordinates of the graph in d dimensions
        triangles : list or set of 3-tuples (v_0, v_1, v_2) corresponding
            to the vertex indices restricting the faces
        """
        self.vertices = np.array(vertices, dtype='float')
        self.dimension = self.vertices.shape[1]
        self.triangles = np.array(triangles)

        if refinement > 0:
            for i in range(refinement):
                self.refine_mesh()

    def refine_mesh(self):
        """
        numpy-array based refinement for a triangular grid. Vertex coordinates
        are stored redundant for each triangle, there is no information about
        connectivity
        #
        #        1
        #       /\
        #      /  \
        #   01/____\12    Construct new triangles
        #    /\    /\       t1 [0,01,02]
        #   /  \  /  \      t2 [01,1,12]
        #  /____\/____\     t3 [02,01,12]
        # 0     02     2    t4 [02,12,2]
        """
        v_0 = self.vertices[self.triangles[:, 0]]
        v_1 = self.vertices[self.triangles[:, 1]]
        v_2 = self.vertices[self.triangles[:, 2]]
        v_02 = (v_0+v_2)*0.5
        v_01 = (v_0+v_1)*0.5
        v_12 = (v_1+v_2)*0.5
        v_02 /= np.sqrt(np.sum(v_02**2, axis=1)).reshape(-1, 1)
        v_01 /= np.sqrt(np.sum(v_01**2, axis=1)).reshape(-1, 1)
        v_12 /= np.sqrt(np.sum(v_12**2, axis=1)).reshape(-1, 1)

        self.vertices = np.array([
            v_0, v_01, v_02,
            v_01, v_1, v_12,
            v_02, v_01, v_12,
            v_02, v_12, v_2])
        self.vertices = self.vertices.swapaxes(0, 1)
        self.vertices = self.vertices.reshape(-1, self.dimension)
        self.triangles = np.arange(len(self.vertices)).reshape((-1, 3))


class DiscreteLaplacian(object):
    def __init__(self, grid, implementation='combinatorial'):
        """

        Parameters
        ----------
        grid: an instance of the Grid class
        implementation: string
            one of 'combinatorial', 'cotan', 'edge-length'
        """

        assert implementation in [
                'combinatorial',
                'cotan',
                'cotan2',
                'edge-length'
                ], 'implementation parameter not recognized'
        self.implementation = implementation

        if implementation == 'combinatorial':
            self.matrix = laplacian(grid.graph)

        if implementation == 'edge-length':
            self.edge_length = grid.graph.new_edge_property('float')
            for e in grid.graph.edges():
                v_0 = grid.graph.vertex_index[e.source()]
                v_1 = grid.graph.vertex_index[e.target()]
                self.edge_length[e] = np.linalg.norm(
                        grid.vertices[v_0] - grid.vertices[v_1])
            self.matrix = laplacian(grid.graph, weight=self.edge_length)

        if implementation == 'cotan':
            """
            For the cotan implementation the weights are assigned by summation
                over the cotan-values corresponding to the angles opposing the
                edge.
            In the representation below the edge e_01 would have the weight
                (1/2) * (cotan(a_01) + cotan(b_01))
            #        v_2
            #       /  \
            #      /a_01\
            #     /      \
            #    /        \
            #   v_0--e_01-v_1
            #    \        /
            #     \      /
            #      \b_01/
            #       \  /
            #        v_3
            We iterate above the trianges, adding the cotans of each angle to
            the weight of the opposing edge
            """
            self.cotan_weight = grid.graph.new_edge_property('float')
            for f in grid.faces:
                v_0, v_1, v_2 = f
                e_01 = grid.graph.edge(v_0, v_1)
                e_02 = grid.graph.edge(v_0, v_2)
                e_12 = grid.graph.edge(v_1, v_2)
                e_01_vector = grid.vertices[v_1] - grid.vertices[v_0]
                e_02_vector = grid.vertices[v_2] - grid.vertices[v_0]
                e_12_vector = grid.vertices[v_2] - grid.vertices[v_1]
                e_01_vector /= np.linalg.norm(e_01_vector)
                e_02_vector /= np.linalg.norm(e_02_vector)
                e_12_vector /= np.linalg.norm(e_12_vector)
                # both vectors need to be reversed here, so we don't need to
                #    multiply by -1
                self.cotan_weight[e_01] += 0.5/np.tan(
                        np.arccos(np.dot(e_02_vector, e_12_vector)))
                # the 01-vector needs to be reversed here, so we multiply by -1
                self.cotan_weight[e_02] += 0.5/np.tan(
                        np.arccos(np.dot(-1*e_01_vector, e_12_vector)))
                self.cotan_weight[e_12] += 0.5/np.tan(
                        np.arccos(np.dot(e_01_vector, e_02_vector)))
            self.matrix = laplacian(grid.graph, weight=self.cotan_weight)

        if implementation == 'cotan2':
            """ the weight w_ij is given by |e*_ij|/|e_ij| where
                |e*_ij| = signed euclidean length of dual edge intersecting
                    edge e_ij
                |e_ij| = euclidean length of edge e_ij
            """
            if not hasattr(grid, 'dual_grid'):
                grid.get_barycentric_dual()

            self.weight = grid.graph.new_edge_property('float')
            for e in grid.graph.edges():
                v_0 = grid.graph.vertex_index[e.source()]
                v_1 = grid.graph.vertex_index[e.target()]
                primal_edge_length = np.linalg.norm(
                        grid.vertices[v_0] - grid.vertices[v_1])
                self.weight[e] = (
                        grid.dual_edge_length[e]/primal_edge_length)

            self.matrix = laplacian(grid.graph, weight=self.weight)

    def calculate_eigensystem(self):
        """
        calculate the eigensystem corresponding to the matrix

        Parameters:
        ----------
        force_real: boolean
            if True, only the real part of the calculated eigensystem will be
            returned. The numerical eigensystem solution may have a non-zero
            imaginary part, while the analytical solution is assured to be real
        """
        if self.implementation in [
                'combinatorial',
                'edge-length',
                'cotan',
                'cotan2']:
            # above implementations lead to symmetric matrices, so we can use
            #    numpy.linalg.eigh (which returns a strictly real eigensystem)
            self.eigensystem = np.linalg.eigh(self.matrix.todense())
        else:
            self.eigensystem = np.linalg.eig(self.matrix.todense())

        return self.eigensystem


class GridPlotter(object):
    def __init__(self, grid, backend='matplotlib'):
        """
        Parameters:
        """
        self.backend = backend
        self.grid = grid
        self.vertices = grid.vertices
        self.faces = np.array(grid.faces)
        self.face_coordinates = [self.vertices[f] for f in self.faces]
        self.cm = plt.get_cmap('seismic')

    def plot_face_values(
            self,
            values,
            layout=None,
            num_subplots=None,
            width=None,
            height=None):
        if layout is None and num_subplots is None:
            num_subplots = 1
            layout = (1, 1)
        elif num_subplots is None:
            num_subplots = layout[0]*layout[1]
        elif layout is None:
            layout = (num_subplots, 1)

        if height is None:
            height = 300*layout[0]

        if width is None:
            width = 300*layout[1]

        if len(values.shape) == 1:
            values = values[np.newaxis, :]

        values -= np.min(values)
        values = values / np.max(values)

        if self.backend == 'matplotlib':
            self.fig = plt.figure(figsize=(3*layout[1], 3*layout[0]))
            for i in range(num_subplots):
                ax = self.fig.add_subplot(
                        layout[0], layout[1], i + 1, projection='3d')
                facecolors = [self.cm(val) for val in values[i]]
                coll = Poly3DCollection(
                    self.face_coordinates,
                    facecolors=facecolors,
                    edgecolors='black')
                ax.add_collection(coll)
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.set_zlim(-1, 1)
                ax.elev = 30

        if self.backend == 'plotly':

            # every polygon will be triangulated for plotting, the facecolors
            #    passed to plotly's Mesh3d need to have the length of the total
            #    triangle count. So we duplicate each face value such that it
            #    appears (n-2) times where n is the polygon order
            values = values.swapaxes(0, 1)
            values = np.vstack(
                    [[values[i]]*(len(self.faces[i])-2)
                        for i in range(len(values))])
            values = values.swapaxes(0, 1)
            print(values.shape)
            if any([len(f) > 3 for f in self.faces]):
                triangles = self.grid.get_triangulation()
            else:
                triangles = self.faces

            fig = plotly.tools.make_subplots(
                    rows=layout[0],
                    cols=layout[1],
                    specs=([[{'is_3d': True}]*layout[1]]*layout[0])
                )
            x, y, z = (
                    self.grid.vertices[:, 0],
                    self.grid.vertices[:, 1],
                    self.grid.vertices[:, 2])
            for i in range(num_subplots):
                facecolors = [self.cm(val) for val in values[i]]
                trisurf = plotly_trisurf(
                            x, y, z,
                            np.array(triangles),
                            facecolors,
                            # plot_edges=1
                            )
                fig.append_trace(
                        trisurf, (i // layout[1]) + 1, (i % layout[1]) + 1)
                # fig.append_trace(edges,
                #     (i // layout[1]) + 1,
                #     (i % layout[1]) + 1)
            fig['layout'].update(
                    height=height,
                    width=width,
                )
            iplot(fig)


class SphericalVoronoiPlotter(object):
    def __init__(self, vertices, radius=1, center=(0, 0, 0)):
        self.radius = radius
        self.center = center
        self.vertices = vertices
        self.cm = plt.get_cmap('seismic')

    def plot_vertex_values(self, values, layout=None, num_subplots=1):
        if layout is None:
            layout = (num_subplots, 1)
        self.fig = plt.figure(figsize=(3*layout[1], 3*layout[0]))
        values = np.array(values)
        if len(values.shape) == 1:
            values = values[np.newaxis, :]

        values -= np.min(values)
        values = values / np.max(values)

        sv = SphericalVoronoi(self.vertices, self.radius, self.center)
        sv.sort_vertices_of_regions()

        for i in range(num_subplots):
            ax = self.fig.add_subplot(
                    layout[0], layout[1], i + 1, projection='3d')
            vertex_colors = [self.cm(val) for val in values[i]]
            for j, region in enumerate(sv.regions):
                color = vertex_colors[j]
                polygon = Poly3DCollection([sv.vertices[region]], alpha=1.0)
                polygon.set_color(color)
                ax.add_collection3d(polygon)

            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            ax.elev = 20


def angles(F, G, compute_vectors=False):
    """
    this function is copied from the krypy module:
    ----------
    Principal angles between two subspaces.
    This algorithm is based on algorithm 6.2 in `Knyazev, Argentati. Principal
    angles between subspaces in an A-based scalar product: algorithms and
    perturbation estimates. 2002.` This algorithm can also handle small angles
    (in contrast to the naive cosine-based svd algorithm).
    :param F: array with ``shape==(N,k)``.
    :param G: array with ``shape==(N,l)``.
    :param compute_vectors: (optional) if set to ``False`` then only the angles
      are returned (default). If set to ``True`` then also the principal
      vectors are returned.
    :return:
      * ``theta`` if ``compute_vectors==False``
      * ``theta, U, V`` if ``compute_vectors==True``
      where
      * ``theta`` is the array with ``shape==(max(k,l),)`` containing the
        principal angles
        :math:`0\\leq\\theta_1\\leq\\ldots\\leq\\theta_{\\max\\{k,l\\}}\\leq
        \\frac{\\pi}{2}`.
      * ``U`` are the principal vectors from F with
        :math:`\\langle U,U \\rangle=I_k`.
      * ``V`` are the principal vectors from G with
        :math:`\\langle V,V \\rangle=I_l`.
    The principal angles and vectors fulfill the relation
    :math:`\\langle U,V \\rangle = \
    \\begin{bmatrix} \
    \\cos(\\Theta) & 0_{m,l-m} \\\\ \
    0_{k-m,m} & 0_{k-m,l-m} \
    \\end{bmatrix}`
    where :math:`m=\\min\\{k,l\\}` and
    :math:`\\cos(\\Theta)=\\operatorname{diag}(\\cos(\\theta_1),\\ldots,\\cos(\\theta_m))`.
    Furthermore,
    :math:`\\theta_{m+1}=\\ldots=\\theta_{\\max\\{k,l\\}}=\\frac{\\pi}{2}`.
    """
    # make sure that F.shape[1]>=G.shape[1]
    reverse = False
    if F.shape[1] < G.shape[1]:
        reverse = True
        F, G = G, F

    QF, _ = scipy.linalg.qr(F, mode='economic')
    QG, _ = scipy.linalg.qr(G, mode='economic')

    # one or both matrices empty? (enough to check G here)
    if G.shape[1] == 0:
        theta = np.ones(F.shape[1])*np.pi/2
        U = QF
        V = QG
    else:
        Y, s, Z = scipy.linalg.svd(np.dot(QF.T.conjugate(), QG))
        Vcos = np.dot(QG, Z.T.conj())
        n_large = np.flatnonzero((s**2) < 0.5).shape[0]
        np.flatnonzero((s**2) < 0.5)
        n_small = s.shape[0] - n_large
        theta = np.r_[
            np.arccos(s[n_small:]),  # [-i:] does not work if i==0
            np.ones(F.shape[1]-G.shape[1])*np.pi/2]
        if compute_vectors:
            Ucos = np.dot(QF, Y)
            U = Ucos[:, n_small:]
            V = Vcos[:, n_small:]

        if n_small > 0:
            RG = Vcos[:, :n_small]
            S = RG - np.dot(QF, np.dot(QF.T.conjugate(), RG))
            _, R = scipy.linalg.qr(S)
            Y, u, Z = scipy.linalg.svd(R)
            theta = np.r_[
                np.arcsin(u[::-1][:n_small]),
                theta]
            if compute_vectors:
                RF = Ucos[:, :n_small]
                Vsin = np.dot(RG, Z.T.conj())
                # next line is hand-crafted since the line from the paper does
                # not seem to work.
                Usin = np.dot(RF, np.dot(
                    np.diag(1/s[:n_small]),
                    np.dot(Z.T.conj(), np.diag(s[:n_small]))))
                U = np.c_[Usin, U]
                V = np.c_[Vsin, V]

    if compute_vectors:
        if reverse:
            U, V = V, U
        return theta, U, V
    else:
        return theta


def principal_angle(a, b):
    """
    primitive version of principal angle calculation for two subspaces
    Parameters:
    ----------
    a, b: numpy arrays with shapes (n,k) and (n,l)
    """
    qa, _ = np.linalg.qr(a)
    qb, _ = np.linalg.qr(b)
    svd = np.linalg.svd(np.dot(np.transpose(qa), qb))
    return np.arccos(min(svd[1].min(), 1.0))


def principal_angle_adv(a, b):
    """
    more advanced / sophisticated version of principal angle calculation,
    using an implementation from krypy
    Parameters:
    ----------
    a, b: numpy arrays with shapes (n,k) and (n,l)
    """
    return np.arccos(np.prod(np.cos(angles(a, b))))


def plotly_trisurf(x, y, z, simplices, facecolors, plot_edges=None):
    # x, y, z are lists of coordinates of the triangle vertices
    #     only triangles atm
    # simplices are the simplices that define the triangularization;
    # simplices  is a numpy array of shape (no_triangles, 3)

    points3D = np.vstack((x, y, z)).T
    tri_vertices = list(map(lambda index: points3D[index], simplices))
    # triangle vertices
    i, j, k = tri_indices(simplices)
    facecolors = np.array(facecolors)
    facecolors *= 255
    facecolors = [
            'rgb({:d},{:d},{:d})'.format(int(c[0]), int(c[1]), int(c[2]))
            for c in facecolors]

    triangles = plotly.graph_objs.Mesh3d(
            x=x,
            y=y,
            z=z,
            facecolor=facecolors,
            i=i,
            j=j,
            k=k,
            # alphahull=0 # convex hull
            lighting=dict(specular=1.0, ),
            )

    if plot_edges is None:  # the triangle sides are not plotted
        return triangles

    else:
        # define the lists Xe, Ye, Ze, of x, y, resp z coordinates of edge end
        #    points for each triangle
        # None separates data corresponding to two consecutive triangles

        lists_coord = [[[
            T[k % 3][c] for k in range(4)] + [None]
            for T in tri_vertices] for c in range(3)]
        Xe, Ye, Ze = [
                reduce(lambda x, y: x + y, lists_coord[k])
                for k in range(3)]

        # define the lines to be plotted
        lines = plotly.graph_objs.Scatter3d(
                x=Xe,
                y=Ye,
                z=Ze,
                mode='lines',
                line=plotly.graph_objs.Line(color='black', width=1.5)
               )
        return triangles, lines
        # return plotly.graph_objs.Data([triangles, lines])


def tri_indices(simplices):
    # simplices is a numpy array defining the simplices of the
    #     triangularization
    # returns the lists of indices i, j, k

    return ([triplet[c] for triplet in simplices] for c in range(3))


def get_orthogonal_matrix(
        parameters,
        dimension,
        method='givens_rotations',
        det=1):
    """
    return an orthogonal nxn matrix parameterized by variables corresponding to
        the degrees of freedom.
    Parameters:
    ----------
    parameters: iterable of (n*(n-1)/2) parameters where n is the dimension of
        the matrix
    dimension: int, dimension of the orthogonal matrix to be parameterized
    method: str corresponding to the method of paramatrization,  one of
        'givens_rotations' or 'matrix_exponential'. In case of
        'givens_rotations' the givens rotation matrices corresponding to the
        parameters are combined, see
            https://en.wikipedia.org/wiki/Givens_rotation
        In case of 'matrix_exponential', a skew symmetric matrix is constructed
        from the parameters and the matrix exponential is applied to the skew
        symmetric matrix, yielding an orthogonal matrix.
        Both methods are surjective w.r.t. SO(n)
    det: int, -1 or 1, corresponding to the desired determinant. The methods
        described above lead to matrices with +1 determinant. To get a
        -1 determinant the first row is negated.

    returns: orthogonal matrix of dimension (nxn) as numpy-array
    """
    assert len(parameters) == (dimension)*(dimension - 1)/2, (
        '{} dimensional orthogonal matrices have {} free parameters, {}'
        ' parameters given'.format(
            dimension, (dimension*(dimension - 1)/2), len(parameters)))
    if method == 'matrix_exponential':
        # build skew symmetric matrices
        A = np.zeros((dimension, dimension))
        parameters_iterator = iter(parameters)
        for i in range(dimension):
            for j in range(dimension):
                if i == j:
                    continue
                if j > i:
                    A[i, j] = next(parameters_iterator)
                else:
                    A[i, j] == -A[j, i]
        U = scipy.linalg.expm(A)
        return U

    if method == 'givens_rotations':
        U = np.eye(dimension)
        parameters_iterator = iter(parameters)
        for i in range(dimension):
            for j in range(dimension):
                if j <= i:
                    continue
                u = next(parameters_iterator)
                G = np.eye(dimension)
                G[i, i], G[j, j] = cos(u), cos(u)
                G[i, j], G[j, i] = sin(u), -sin(u)
                U = np.dot(U, G)
    if det == -1:
        # negating first row corresponds to a multiplication with
        #    a [-1, 1, 1, ...] diagonal matrix
        U[0] = -U[0]

    return U


def non_diagonality_cost(matrix):
    """
    a cost function representing the non-diagonality of a given matrix via
    Parameters:
    ----------
    matrix: np.array or np.matrix
    """
    cost = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i == j:
                continue
            cost += matrix[i, j]**2 / (matrix[i, i] * matrix[j, j] + 1)

    return cost


def get_S(u, U1_dim, U2_dim, U1_det=1, U2_det=1):
    """
    get an orthogonal matrix S of dimension nxn by combining two orthogonal
        matrices U1, U2 of lower dimension constructed by a parameterization
        for each
    Parameters:
    ----------
    u: list or array containing the parameters to construct the orthogonal
        matrices U1 and U2 from, length needs to be
            (U1_dim*(U1_dim-1)/2) + (U2_dim*(U2_dim-1)/2)
    U1_dim: int, specifying dimension of U1
    U2_dim: int, specifying dimension of U2
    U1_det: int, specifying determinant of U1
    U2_det: int, specifying determinant of U2
    """

    # a n-dimensional orthogonal matrix has n*(n-1)/2 degrees of freedom
    U1_dof = U1_dim*(U1_dim - 1)//2

    # split paramater array into parameters corresponding to U1 and U2
    u1 = u[:U1_dof]
    u2 = u[U1_dof:]

    # get orthogonal matrices U1, U2 by parameter arrays
    U1 = get_orthogonal_matrix(u1, U1_dim, det=U1_det)
    U2 = get_orthogonal_matrix(u2, U2_dim, det=U2_det)

    # build S by "probing" the combination of U1 and U2 with unit vectors
    dim = U1_dim*U2_dim
    S = np.zeros((dim, dim))
    for i in range(dim):
        S[i] = (U2@
                (U1@
                 (np.r_[[0]*i, 1, [0]*(dim-1-i)].reshape((U2_dim, U1_dim))
                  ).T
                 ).T
                ).T.reshape(dim)

    return S


def get_L_twiddle_cost(u, L, U1_dim, U2_dim, U1_det=1, U2_det=1):
    """
    wrapper function intended to being handed over the minimizer. First an
        orthogonal transformation matrix S is being created by combining to
        orthogonal matrices U1, U2 of lower dimension. S is then applied to L
        via S@L@S.T and the cost corresponding to non-diagonality is being
        returned
    """
    S = get_S(u, U1_dim, U2_dim, U1_det=1, U2_det=1)
    L_twiddle = S@L@(S.T)
    L_twiddle_cost = non_diagonality_cost(L_twiddle)
    return L_twiddle_cost
