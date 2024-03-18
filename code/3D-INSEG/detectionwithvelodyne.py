import plotly.graph_objects as go
import numpy as np
from pathlib import Path
from imageio import imread
import matplotlib.pyplot as plt
import os

def getindexes(mask):
    xp=[]
    yp=[]
    for iy, ix in np.ndindex(mask.shape):
        if (mask[iy, ix]==True):
            xp.append(ix)
            yp.append(iy)
    return np.asarray(xp),np.asarray(yp)

def getInferencepoints(masks,disp):
    
    #location set in pixels indexes
    XP=[]
    YP=[]


    for mask in masks:
        xp,yp=getindexes(mask)

        XP.append(xp)
        YP.append(yp)

    #XP=np.asarray(XP)
    #YP=np.asarray(YP)


    Z= []
    for i in range(len(XP)):
        z_inf=np.array([])
        for xp, yp in zip(XP[i],YP[i]):
            z_inf=np.append(z_inf,disp[int(yp)][int(xp)])
        Z.append(z_inf)

    #set of locations
    XL= XP
    YL= YP
    #Z=np.array(Z)

    P=[]
    for i in range(len(XL)):
        points_inf = np.stack((XL[i], YL[i], Z[i]), axis=0) 
        #points_inf[0]=points_inf[0]-baseline/2
        P.append(points_inf)

    return P
def create_figure():

    fig = go.Figure(
    data=[
    ],
    layout=dict(
        scene=dict(
            xaxis=dict(visible=True),
            yaxis=dict(visible=True),
            zaxis=dict(visible=True),
            #Make the grid equal on all axes
            aspectratio=dict(x=1, y=1, z=1),
        )
    )
)
    return fig

def InverseProjection(P):
    Inverseprojections=[]
    for i in range(len(P)):
        XP,YP,disp=P[i]

        #DEFINE THE CAM CALIB
        # Camera intrinsic parameters (focal length, principal point)
        fx, fy = 342, 342
        cx, cy = 308, 183

        #fx, fy, cx, cy = 689.931, 689.931, 620.583, 360.990
        
        # Define the camera matrix
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        # Baseline distance between cameras (for a stereo camera setup)
        baseline = 120

        # Pixel coordinates and disparity of points in the image
        us = XP
        vs = YP
        ds = disp

        # Convert disparities to depths
        focal_length_px = (fx + fy) / 2  # assuming fx = fy
        Zs = (focal_length_px * baseline) / ds

        # Convert pixel coordinates and depths to 3D points
        uv_1 = np.vstack((us, vs, np.ones_like(us)))
        K_inv = np.linalg.inv(K)
        X_cams = np.dot(K_inv, uv_1) * Zs
        X_cams = np.vstack((X_cams, np.ones_like(X_cams[0])))

        Inverseprojections.append(X_cams)

    return Inverseprojections

def load_velodynepoints(velodynepoints_paths,thr=100):
    Xvel=np.load(velodynepoints_paths[0])
    Yvel=np.load(velodynepoints_paths[1])
    Zvel=np.load(velodynepoints_paths[2])

    # Modify arrays
    Xvel = Xvel[0:].T[0]
    Yvel = Yvel[0:].T[0]
    Zvel = Zvel[0:].T[0]

    # Compute Euclidean distance from origin
    try:
        distance = np.sqrt(Xvel**2 + Yvel**2 + Zvel**2)
    

        # Filter points within a certain distance from the origin
        threshold = thr
        filtered_indices = np.where(distance <= threshold)
        filtered_X = Xvel[filtered_indices]
        filtered_Y = Yvel[filtered_indices]
        filtered_Z = Zvel[filtered_indices]
    except:
        filtered_X = Xvel
        filtered_Y = Yvel
        filtered_Z = Zvel
    

    return filtered_X,filtered_Y,filtered_Z

#create function.
def visualize2dposition(images_folder_path,disparity_path,inferences_folder_path,velodynepoints_paths,save_path,use_velodyne=True,save_html=False):
    
    testF_folder=Path(images_folder_path)
    #load disparty image
    disp = np.load(disparity_path)
    #load image
    image = imread(testF_folder / "im0.jpg")
    #load inferences
    inf_path=Path(inferences_folder_path)
    masks=np.load(inf_path/f"masks.npy",allow_pickle=True)
    scores=np.load(inf_path/f"scores.npy",allow_pickle=True)
    labels=np.load(inf_path/f"labels.npy",allow_pickle=True)

    #save inferences points
    thr=0.9
    masks_thr=masks[np.where(scores>thr)]
    labels_thr=labels[np.where(scores>thr)]

    #get inference points
    P=getInferencepoints(masks_thr,disp)

    #create figure matplotli
    fig = plt.figure()


    #Do the inverse projection
    Inferences_points=InverseProjection(P)


    for i in range(len(Inferences_points)):

        x_centroid=np.mean(-Inferences_points[i][0]-270)
        y_centroid=np.mean(Inferences_points[i][1]-300)
        z_centroid=np.mean(Inferences_points[i][2])

        #add centroid to the figure
        plt.scatter(x_centroid,y_centroid,s=100,c='r',marker='x')

    

    #load velodyne points
    if use_velodyne:
        X_vel,Y_vel,Z_vel=load_velodynepoints(velodynepoints_paths)

        #add velodyne points to the figure
        Points=        go.Scatter3d(
                x=-1000*Y_vel, y=1000*Z_vel, z=-1000*X_vel, # flipped to make visualization nicer
                mode='markers',
                marker=dict(size=1, color='blue',opacity=1),
                name=f'Point cloud'
            )
        fig.add_trace(Points)

    #save figure in html format
    if save_html:
        fig.write_html(save_path)

    return fig,P

def visualize_pointcloud(images_folder_path,disparity_path,inferences_folder_path,velodynepoints_paths,save_path,use_velodyne=True,save_html=False,thr=0.9,filter_label=False,labels_to_visualize=[0],objects_index=[0]):

    testF_folder=Path(images_folder_path)
    #load disparty image
    disp = np.load(disparity_path)
    #load image
    image = imread(testF_folder / "im0.jpg")
    #load inferences
    inf_path=Path(inferences_folder_path)
    masks=np.load(inf_path/f"masks.npy",allow_pickle=True)
    scores=np.load(inf_path/f"scores.npy",allow_pickle=True)
    labels=np.load(inf_path/f"labels.npy",allow_pickle=True)

    #save inferences points
    thr=thr
    masks_thr=masks[np.where(scores>thr)]
    labels_thr=labels[np.where(scores>thr)]

    if filter_label:
        masks_thr=masks_thr[np.where(labels_thr==labels_to_visualize)]
        labels_thr=labels_thr[np.where(labels_thr==labels_to_visualize)]

    #get inference points
    try:
        P=getInferencepoints(masks_thr[objects_index],disp)
    except:
        P=getInferencepoints(masks_thr,disp)
    #create figure
    fig=create_figure()

    #Do the inverse projection
    Inferences_points=InverseProjection(P)

    points_data=[]

    for i in range(len(Inferences_points)):

        Points=        go.Scatter3d(
                    x=-Inferences_points[i][0]-270, y=Inferences_points[i][1]-300, z=Inferences_points[i][2], # not flipped to make visualization nicer
                    mode='markers',
                    marker=dict(size=1, color=int(labels_thr[i]),opacity=0.1),
                    name=f'objeto{i},tipo:{labels_thr[i]}'
                )
        x_centroid=np.mean(-Inferences_points[i][0]-270)
        y_centroid=np.mean(Inferences_points[i][1]-300)
        z_centroid=np.mean(Inferences_points[i][2])
        centroids=        go.Scatter3d(
                    x=[x_centroid], y=[y_centroid], z=[z_centroid], # not flipped to make visualization nicer
                    mode='markers',
                    marker=dict(size=5, color='black',opacity=1),
                    name=f'objeto{i},centroid'
                )
                
        fig.add_trace(Points)
        fig.add_trace(centroids)

        points_data.append([-Inferences_points[i][0]-270,Inferences_points[i][1]-300,Inferences_points[i][2]])

    #load velodyne points
    if use_velodyne:
        X_vel,Y_vel,Z_vel=load_velodynepoints(velodynepoints_paths)

        #add velodyne points to the figure
        Points=        go.Scatter3d(
                x=-1000*Y_vel, y=1000*Z_vel, z=-1000*X_vel, # flipped to make visualization nicer
                mode='markers',
                marker=dict(size=1, color='blue',opacity=1),
                name=f'Point cloud'
            )
        fig.add_trace(Points)

    #save figure in html format
    if save_html:
        fig.write_html(save_path)

    return fig,P,points_data


