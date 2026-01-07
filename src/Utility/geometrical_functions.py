import numpy as np
def cartesian_to_cylindrical(cartesian_point):
    x, y, z = cartesian_point
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    return [r, theta, z]


def cylindrical_to_cartesian(cylindrical_point):
    r, theta, z = cylindrical_point
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return [x, y, z]


def get_node_normals(surface_mesh):
    node_normals = np.zeros_like(surface_mesh)
    for j in range(surface_mesh.shape[0]):
        for i in range(surface_mesh.shape[1]):

            if j == int(surface_mesh.shape[0] - 1):
                if i == int(surface_mesh.shape[1] - 1):
                    v0 = surface_mesh[j, i] - surface_mesh[j, i - 1]
                    v1 = surface_mesh[j, i] - surface_mesh[j - 1, i]
                    direction = 1
                else:
                    v0 = surface_mesh[j, i] - surface_mesh[j, i + 1]
                    v1 = surface_mesh[j, i] - surface_mesh[j - 1, i]
                    direction = -1

            elif i == int(surface_mesh.shape[1] - 1) and j != int(surface_mesh.shape[0] - 1):
                v0 = surface_mesh[j, i] - surface_mesh[j, i - 1]
                v1 = surface_mesh[j, i] - surface_mesh[j + 1, i]
                direction = -1

            else:
                v0 = surface_mesh[j, i] - surface_mesh[j, i + 1]
                v1 = surface_mesh[j, i] - surface_mesh[j + 1, i]
                direction = 1

            dummy_normal = direction * np.cross(v0, v1)
            length = np.sqrt(np.sum(np.power(dummy_normal, 2)))
            node_normals[j, i] = - dummy_normal / length

    return node_normals


def get_mag_of_vec(Point):
    return np.sqrt(Point[0] ** 2 + Point[1] ** 2 + Point[2] ** 2)

