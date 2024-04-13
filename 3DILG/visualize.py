import numpy as np
import meshplot as mp
import os
import trimesh


filenames = os.listdir('/public/home/zhaoyq/ATISS/3DILG/tested_4000/')

for i in range(len(filenames)):
    for j in range(i+1,len(filenames)):
        mesh1 = trimesh.load(os.path.join('/public/home/zhaoyq/ATISS/3DILG/tested_4000', filenames[i]))
        mesh2 = trimesh.load(os.path.join('/public/home/zhaoyq/ATISS/3DILG/tested_4000', filenames[j]))
        u = filenames[i].split('.')[0]
        v = filenames[j].split('.')[0]

        mix_mesh1 = trimesh.load(os.path.join('/public/home/huhzh/zhaoyq/3DILG/mixture', f'{u}_{v}.obj'))
        mix_mesh2 = trimesh.load(os.path.join('/public/home/huhzh/zhaoyq/3DILG/mixture', f'{v}_{u}.obj'))


        mp.website()
        # import ipdb
        # ipdb.set_trace()
        # plot = mp.plot(np.array([[1.0, 1., 0.1]]), return_plot=True)
        sp_recon_mesh = mesh1.simplify_quadratic_decimation(mesh1.faces.shape[0]//500)
        plot = mp.plot(mesh1.vertices, mesh1.faces, c=np.random.rand(*mesh1.faces.shape), return_plot=True)
        # plot.add_mesh(mesh1.vertices, mesh1.faces, c=np.random.rand(*mesh1.faces.shape))
        # plot.add_mesh(mesh2.vertices+5, mesh2.faces, c=np.random.rand(*mesh2.faces.shape))
        # plot.add_mesh(mix_mesh1.vertices+10, mix_mesh1.faces, c=np.random.rand(*mix_mesh1.faces.shape))
        # plot.add_mesh(mix_mesh2.vertices+15, mix_mesh2.faces, c=np.random.rand(*mix_mesh2.faces.shape))
        plot.save(f"{u}_{v}_visualize.html")
        break
    break