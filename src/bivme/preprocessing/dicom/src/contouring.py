import numpy as np
import cv2

def find_center(p1, p2):
    
    """ finds the centroid of two points """

    return [ int((p1[1] + p2[1])/2), int((p1[0] + p2[0])/2 )]

def get_intersections(point_list1, point_list2, distance_cutoff=4.5):

    """ Finds the points that are within a given cutoff distance between two lists """

    a = range(len(point_list1))
    b = range(len(point_list2))
    [A, B] = np.meshgrid(a,b)
    c = np.concatenate([A.T, B.T], axis=0)
    pairs = c.reshape(2,-1).T
    dist = np.sqrt( ( ( point_list1[pairs[:,0],0] - point_list2[pairs[:,1],0] ) ** 2 ) + 
                    ( ( point_list1[pairs[:,0],1] - point_list2[pairs[:,1],1] ) ** 2 ) )
    
    pairs = pairs[np.where(dist < distance_cutoff)[0].tolist()]

    return pairs

def get_valve_points_from_intersections(segmentation, endolabel, superlabel, distance_cutoff=1):
    ## TODO: Pass in contours not segmentation
    endo = (segmentation == endolabel).astype(np.uint8)
    superior = (segmentation == superlabel).astype(np.uint8)
    # Get contours
    contours, hierarchy = cv2.findContours(cv2.inRange(endo, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        endopts = []
        for c in contours:
            endopts += [x.tolist() for i,x in enumerate(c[:, 0, :]) if i % 2 == 0]
        endopts = np.array(endopts, dtype=np.int64)
    else:
        endopts = []

    contours, hierarchy = cv2.findContours(cv2.inRange(superior, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        suppts = []
        for c in contours:
            suppts += [x.tolist() for i,x in enumerate(c[:, 0, :]) if i % 2 == 0]
        suppts = np.array(suppts, dtype=np.int64)
    else:
        suppts = []
    
    # Get intersection points between endo and superior
    while distance_cutoff < 10:
        pairs = get_intersections(endopts, suppts, distance_cutoff)
        if len(pairs) > 2:
            break
        else:
            distance_cutoff += 0.5

    # Valve points will be extents of the intersection points
    valveplane = suppts[pairs[:,1],:]
    # Find extent
    x = [v[0] for v in valveplane]
    y = [v[1] for v in valveplane]
    center = [np.mean(x), np.mean(y)]
    distances = [np.sqrt((v[0] - center[0])**2 + (v[1] - center[1])**2) for v in valveplane]
    valveplane = valveplane[np.argsort(distances)]
    radius = int(np.max(distances))
    
    max_dist = radius
    for i in range(len(valveplane)-len(valveplane//2), len(valveplane)):
        for j in range(i+1, len(valveplane)):
            dist = np.sqrt((valveplane[i][0] - valveplane[j][0])**2 + (valveplane[i][1] - valveplane[j][1])**2)
            if dist > radius:
                if dist > max_dist:
                    max_dist = dist
                    valvepts = np.array([valveplane[i], valveplane[j]], dtype=np.int64)

    return valvepts

def estimate_lva(epipts, mv1, mv2):
    mv_centroid = [np.mean([mv1[0], mv2[0]]), np.mean([mv1[1], mv2[1]])]
    distances = [np.sqrt((p[0] - mv_centroid[0])**2 + (p[1] - mv_centroid[1])**2) for p in epipts]
    lvepiapex = epipts[np.argmax(distances)]
    return lvepiapex

def contour_SAX(segmentation):
    # extract points
    LV_endo = (segmentation == 1).astype(np.uint8)
    LV_myo = (segmentation == 2).astype(np.uint8)
    LV_epi = (LV_endo | LV_myo).astype(np.uint8)
    RV_endo = (segmentation == 3).astype(np.uint8)
    RV_myo = (segmentation == 4).astype(np.uint8)
    RV_epi = (RV_endo | RV_myo).astype(np.uint8)

    # convert to contours
    contours, hierarchy = cv2.findContours(cv2.inRange(LV_endo, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        LV_endo_pts = np.array([x.tolist() for i,x in enumerate(c[:, 0, :]) if i % 2 == 0], dtype=np.int64)
    else:
        LV_endo_pts = []

    contours, hierarchy = cv2.findContours(cv2.inRange(LV_epi, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        LV_epi_pts = np.array([x.tolist() for i,x in enumerate(c[:, 0, :]) if i % 2 == 0], dtype=np.int64)
    else:
        LV_epi_pts = []

    contours, hierarchy = cv2.findContours(cv2.inRange(LV_myo, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        LV_myo_pts = []
        for c in contours:
            LV_myo_pts += [x.tolist() for i,x in enumerate(c[:, 0, :]) if i % 2 == 0]
        LV_myo_pts = np.array(LV_myo_pts, dtype=np.int64)
    else:
        LV_myo_pts = []

    contours, hierarchy = cv2.findContours(cv2.inRange(RV_endo, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        RV_endo_pts = np.array([x.tolist() for i,x in enumerate(c[:, 0, :]) if i % 2 == 0], dtype=np.int64)
    else:
        RV_endo_pts = []

    contours, hierarchy = cv2.findContours(cv2.inRange(RV_epi, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        RV_epi_pts = []
        for c in contours:
            RV_epi_pts += [x.tolist() for i,x in enumerate(c[:, 0, :]) if i % 2 == 0]
        RV_epi_pts = np.array(RV_epi_pts, dtype=np.int64)
    else:
        RV_epi_pts = []

    contours, hierarchy = cv2.findContours(cv2.inRange(RV_myo, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        RV_myo_pts = []
        for c in contours:
            RV_myo_pts += [x.tolist() for i,x in enumerate(c[:, 0, :]) if i % 2 == 0]
        RV_myo_pts = np.array(RV_myo_pts, dtype=np.int64)
    else:
        RV_myo_pts = []

    # Get intersection points between RV endo and LV epi to separate septal wall from free wall
    if len(RV_endo_pts)>0 and len(LV_epi_pts)>0:
        pairs = get_intersections(RV_endo_pts, LV_epi_pts, distance_cutoff=3)

        if len(pairs) > 0:
            RV_septal_pts = RV_endo_pts[np.unique(pairs[:,0])] # deletes intersection from RV endo pts
            RV_fw_pts = np.array([pnt.tolist() for i, pnt in enumerate(RV_endo_pts) if i not in np.unique(pairs[:,0])], 
                                    dtype=np.int64)
            LV_epi_pts = np.array([pnt.tolist() for i, pnt in enumerate(LV_epi_pts) if i not in np.unique(pairs[:,1])], 
                                    dtype=np.int64)
        else:
            RV_septal_pts = []
            RV_fw_pts = RV_endo_pts
    else:
        RV_septal_pts = []
        RV_fw_pts = RV_endo_pts
    

    # # Get intersection points between RV epi and RV endo to remove extraneous RV epi points
    # if len(RV_epi_pts)>0 and len(RV_fw_pts)>0:
    #     pairs = get_intersections(RV_epi_pts, RV_fw_pts, distance_cutoff=5)

    #     if len(pairs) > 0:
    #         RV_epi_pts = RV_epi_pts[np.unique(pairs[:,0])]
        
        
    # Get intersection points between RV epi and RV myo to keep only free wall points
    if len(RV_epi_pts)>0 and len(RV_myo_pts)>0:
        pairs = get_intersections(RV_epi_pts, RV_myo_pts, distance_cutoff=1.5)

        if len(pairs) > 0:
            RV_epi_pts = RV_epi_pts[np.unique(pairs[:,0])]


    # Get intersection points between LV epi and LV myo to keep only myocardial points
    if len(LV_epi_pts)>0 and len(LV_myo_pts)>0:
        pairs = get_intersections(LV_epi_pts, LV_myo_pts, distance_cutoff=1.5)

        if len(pairs) > 0:
            LV_epi_pts = LV_epi_pts[np.unique(pairs[:,0])]

    # Remove intersection between RV epi and LV epi
    if len(RV_epi_pts)>0 and len(LV_epi_pts)>0:
        pairs = get_intersections(RV_epi_pts, LV_epi_pts, distance_cutoff=1.5)

        if len(pairs) > 0:
            RV_epi_pts = np.array([pnt.tolist() for i, pnt in enumerate(RV_epi_pts) if i not in np.unique(pairs[:,0])], 
                                dtype=np.int64)
            
    # If there are no lv endo points, remove the lv epi points
    if len(LV_endo_pts) == 0:
        LV_epi_pts = []
                
    return [LV_endo_pts, LV_epi_pts, RV_septal_pts, RV_fw_pts, RV_epi_pts]

def contour_RVOT(segmentation):
    RV_endo = (segmentation == 1).astype(np.uint8)
    RV_myo = (segmentation == 2).astype(np.uint8)
    RV_epi = (RV_endo | RV_myo).astype(np.uint8)
    pa = (segmentation == 3).astype(np.uint8)

    # convert to contours
    contours, hierarchy = cv2.findContours(cv2.inRange(RV_endo, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        RV_endo_pts = np.array([x.tolist() for i,x in enumerate(c[:, 0, :]) if i % 2 == 0], dtype=np.int64)
    else:
        RV_endo_pts = []

    contours, hierarchy = cv2.findContours(cv2.inRange(RV_epi, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        RV_epi_pts = []
        for c in contours:
            RV_epi_pts += [x.tolist() for i,x in enumerate(c[:, 0, :]) if i % 2 == 0]
        RV_epi_pts = np.array(RV_epi_pts, dtype=np.int64)
    else:
        RV_epi_pts = []

    contours, hierarchy = cv2.findContours(cv2.inRange(RV_myo, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        RV_myo_pts = []
        for c in contours:
            RV_myo_pts += [x.tolist() for i,x in enumerate(c[:, 0, :]) if i % 2 == 0]
        RV_myo_pts = np.array(RV_myo_pts, dtype=np.int64)
    else:
        RV_myo_pts = []

    contours, hierarchy = cv2.findContours(cv2.inRange(pa, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        pa_pts = np.array([x.tolist() for i,x in enumerate(c[:, 0, :]) if i % 2 == 0], dtype=np.int64)
    else:
        pa_pts = []

    # Get intersection points between RV endo and pa to clean RV endo pts
    if len(RV_endo_pts)>0 and len(pa_pts)>0:
        pairs = get_intersections(RV_endo_pts, pa_pts, distance_cutoff=1.5)

        if len(pairs) > 0:
            RV_endo_pts = np.array([pnt.tolist() for i, pnt in enumerate(RV_endo_pts) if i not in np.unique(pairs[:,0])], 
                                dtype=np.int64)
            
    # Get intersection points between RV epi and pa to clean RV epi pts
    if len(RV_epi_pts)>0 and len(pa_pts)>0:
        pairs = get_intersections(RV_epi_pts, pa_pts, distance_cutoff=1.5)

        if len(pairs) > 0:
            RV_epi_pts = np.array([pnt.tolist() for i, pnt in enumerate(RV_epi_pts) if i not in np.unique(pairs[:,0])], 
                                dtype=np.int64)

            
    # Get intersection points between RV myo and RV endo to separate free wall from septal wall
    if len(RV_endo_pts)>0 and len(RV_myo_pts)>0:

        pairs = get_intersections(RV_endo_pts, RV_myo_pts, distance_cutoff = 3)

        if len(pairs) > 0:
            RV_fw_pts = np.array([pnt.tolist() for i, pnt in enumerate(RV_endo_pts) if i in np.unique(pairs[:,0])], 
                                    dtype=np.int64)
            RV_s_pts = np.array([pnt.tolist() for i, pnt in enumerate(RV_endo_pts) if i not in np.unique(pairs[:,0])],
                                    dtype=np.int64)
            
        else:
            RV_fw_pts = RV_endo_pts
            RV_s_pts = []

    else:
        RV_fw_pts = RV_endo_pts
        RV_s_pts = []
            
    # # Get intersection points between RV epi and RV endo to remove extraneous RV epi points
    # if len(RV_epi_pts)>0 and len(RV_fw_pts)>0:
    #     pairs = get_intersections(RV_epi_pts, RV_fw_pts, distance_cutoff=5)

    #     if len(pairs) > 0:
    #         RV_epi_pts = RV_epi_pts[np.unique(pairs[:,0])]

    # Get intersection points between RV epi and RV myo pts keep only free wall points
    if len(RV_epi_pts)>0 and len(RV_myo_pts)>0:
        pairs = get_intersections(RV_epi_pts, RV_myo_pts, distance_cutoff=1.5)

        if len(pairs) > 0:
            RV_epi_pts = RV_epi_pts[np.unique(pairs[:,0])]


    return [RV_s_pts, RV_fw_pts, RV_epi_pts, pa_pts]


def contour_2ch(segmentation):
    # extract points
    LV_endo = (segmentation == 1).astype(np.uint8)
    LV_myo = (segmentation == 2).astype(np.uint8)
    LV_epi = (LV_endo | LV_myo).astype(np.uint8)
    la = (segmentation == 3).astype(np.uint8)

    # convert to contours
    contours, hierarchy = cv2.findContours(cv2.inRange(LV_endo, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        LV_endo_pts = np.array([x.tolist() for i,x in enumerate(c[:, 0, :]) if i % 2 == 0], dtype=np.int64)
    else:
        LV_endo_pts = []

    contours, hierarchy = cv2.findContours(cv2.inRange(LV_epi, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        LV_epi_pts = np.array([x.tolist() for i,x in enumerate(c[:, 0, :]) if i % 2 == 0], dtype=np.int64)
    else:
        LV_epi_pts = []

    contours, hierarchy = cv2.findContours(cv2.inRange(la, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        la_pts = np.array([x.tolist() for i,x in enumerate(c[:, 0, :]) if i % 2 == 0], dtype=np.int64)
    else:
        la_pts = []

    # Get intersection points between LV endo and LA to clean LV endo pts 
    if len(LV_endo_pts)>0 and len(la_pts)>0:
        pairs = get_intersections(LV_endo_pts, la_pts, distance_cutoff=1.5)

        if len(pairs) > 0:
            LV_endo_pts = np.array([pnt.tolist() for i, pnt in enumerate(LV_endo_pts) if i not in np.unique(pairs[:,0])], 
                                dtype=np.int64)
            
    # Get intersection points between LV epi and LA to clean LV epi pts
    if len(LV_epi_pts)>0 and len(la_pts)>0:
        pairs = get_intersections(LV_epi_pts, la_pts, distance_cutoff=1.5)

        if len(pairs) > 0:
            LV_epi_pts = np.array([pnt.tolist() for i, pnt in enumerate(LV_epi_pts) if i not in np.unique(pairs[:,0])], 
                                dtype=np.int64)
            
    return [LV_endo_pts, LV_epi_pts, la_pts]

def contour_3ch(segmentation):
    # extract points
    LV_endo = (segmentation == 1).astype(np.uint8)
    LV_myo = (segmentation == 2).astype(np.uint8)
    LV_epi = (LV_endo | LV_myo).astype(np.uint8)
    RV_endo = (segmentation == 3).astype(np.uint8)
    la = (segmentation == 4).astype(np.uint8)
    aorta = (segmentation == 5).astype(np.uint8)
    RV_myo = (segmentation == 6).astype(np.uint8)
    RV_epi = (RV_endo | RV_myo).astype(np.uint8)

    # convert to contours
    # left ventricle
    contours, hierarchy = cv2.findContours(cv2.inRange(LV_endo, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        LV_endo_pts = np.array([x.tolist() for i,x in enumerate(c[:, 0, :]) if i % 2 == 0], dtype=np.int64)
    else:
        LV_endo_pts = [] 

    contours, hierarchy = cv2.findContours(cv2.inRange(LV_epi, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        LV_epi_pts = np.array([x.tolist() for i,x in enumerate(c[:, 0, :]) if i % 2 == 0], dtype=np.int64)
    else:
        LV_epi_pts = []

    # right ventricle
    contours, hierarchy = cv2.findContours(cv2.inRange(RV_endo, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        RV_endo_pts = np.array([x.tolist() for i,x in enumerate(c[:, 0, :]) if i % 2 == 0], dtype=np.int64)
    else:
        RV_endo_pts = []

    contours, hierarchy = cv2.findContours(cv2.inRange(RV_epi, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        RV_epi_pts = []
        for c in contours:
            RV_epi_pts += [x.tolist() for i,x in enumerate(c[:, 0, :]) if i % 2 == 0]
        RV_epi_pts = np.array(RV_epi_pts, dtype=np.int64)
    else:
        RV_epi_pts = []

    contours, hierarchy = cv2.findContours(cv2.inRange(RV_myo, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        RV_myo_pts = []
        for c in contours:
            RV_myo_pts += [x.tolist() for i,x in enumerate(c[:, 0, :]) if i % 2 == 0]
        RV_myo_pts = np.array(RV_myo_pts, dtype=np.int64)
    else:
        RV_myo_pts = []

    # la
    contours, hierarchy = cv2.findContours(cv2.inRange(la, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        la_pts = np.array([x.tolist() for i,x in enumerate(c[:, 0, :]) if i % 2 == 0], dtype=np.int64)
    else:
        la_pts = []
    
    # aorta
    contours, hierarchy = cv2.findContours(cv2.inRange(aorta, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        aorta_pts = np.array([x.tolist() for i,x in enumerate(c[:, 0, :]) if i % 2 == 0], dtype=np.int64)
    else:
        aorta_pts = []

    # Get intersection points between RV endo and LV epi to separate septal wall from free wall
    if len(RV_endo_pts)>0 and len(LV_epi_pts)>0:

        pairs = get_intersections(RV_endo_pts, LV_epi_pts, distance_cutoff=1.5) 

        if len(pairs) > 0:
            RV_septal_pts = RV_endo_pts[np.unique(pairs[:,0])] # deletes intersection from RV endo pts
            RV_fw_pts = np.array([pnt.tolist() for i, pnt in enumerate(RV_endo_pts) if i not in np.unique(pairs[:,0])], 
                                dtype=np.int64)
            
            LV_epi_pts = np.array([pnt.tolist() for i, pnt in enumerate(LV_epi_pts) if i not in np.unique(pairs[:,1])], 
                                dtype=np.int64)
        else:
            RV_septal_pts = []
            RV_fw_pts = RV_endo_pts
    else:
        RV_septal_pts = []
        RV_fw_pts = RV_endo_pts
            
    # # Get intersection points between RV epi and RV endo to remove extraneous RV epi points
    # if len(RV_epi_pts)>0 and len(RV_fw_pts)>0:
    #     pairs = get_intersections(RV_epi_pts, RV_fw_pts, distance_cutoff=5)

    #     if len(pairs) > 0:
    #         RV_epi_pts = RV_epi_pts[np.unique(pairs[:,0])]

    # Get intersection points between LV endo and la to clean LV endo pts
    if len(LV_endo_pts)>0 and len(la_pts)>0:
        pairs = get_intersections(LV_endo_pts, la_pts, distance_cutoff=1.5)

        if len(pairs) > 0:
            LV_endo_pts = np.array([pnt.tolist() for i, pnt in enumerate(LV_endo_pts) if i not in np.unique(pairs[:,0])], 
                                dtype=np.int64)
            
    # Get intersection points between LV epi and la to clean LV epi pts
    if len(LV_epi_pts)>0 and len(la_pts)>0:
        pairs = get_intersections(LV_epi_pts, la_pts, distance_cutoff=1.5)

        if len(pairs) > 0:
            LV_epi_pts = np.array([pnt.tolist() for i, pnt in enumerate(LV_epi_pts) if i not in np.unique(pairs[:,0])], 
                                dtype=np.int64)
    
    # Get intersection points between LV endo and aorta to clean LV endo pts
    if len(LV_endo_pts)>0 and len(aorta_pts)>0:
        pairs = get_intersections(LV_endo_pts, aorta_pts, distance_cutoff=1.5)

        if len(pairs) > 0:
            LV_endo_pts = np.array([pnt.tolist() for i, pnt in enumerate(LV_endo_pts) if i not in np.unique(pairs[:,0])], 
                                dtype=np.int64)
            
    # Get intersection points between LV epi and aorta to clean LV epi pts
    if len(LV_epi_pts)>0 and len(aorta_pts)>0:
        pairs = get_intersections(LV_epi_pts, aorta_pts, distance_cutoff=1.5)

        if len(pairs) > 0:
            LV_epi_pts = np.array([pnt.tolist() for i, pnt in enumerate(LV_epi_pts) if i not in np.unique(pairs[:,0])], 
                                dtype=np.int64)
            
    # Get intersection points between RV epi and RV myo to keep only free wall points
    if len(RV_epi_pts)>0 and len(RV_myo_pts)>0:
        pairs = get_intersections(RV_epi_pts, RV_myo_pts, distance_cutoff=1.5)

        if len(pairs) > 0:
            RV_epi_pts = RV_epi_pts[np.unique(pairs[:,0])]

    
    # Remove intersection between RV epi and LV epi
    if len(RV_epi_pts)>0 and len(LV_epi_pts)>0:
        pairs = get_intersections(RV_epi_pts, LV_epi_pts, distance_cutoff=1.5)

        if len(pairs) > 0:
            RV_epi_pts = np.array([pnt.tolist() for i, pnt in enumerate(RV_epi_pts) if i not in np.unique(pairs[:,0])], 
                                dtype=np.int64)
            
    return [LV_endo_pts, LV_epi_pts, RV_septal_pts, RV_fw_pts, la_pts, aorta_pts, RV_epi_pts]

def contour_4ch(segmentation):
    # extract points
    LV_endo = (segmentation == 1).astype(np.uint8)
    LV_myo = (segmentation == 2).astype(np.uint8)
    LV_epi = (LV_endo | LV_myo).astype(np.uint8)
    RV_endo = (segmentation == 3).astype(np.uint8)
    la = (segmentation == 4).astype(np.uint8)
    ra = (segmentation == 5).astype(np.uint8)
    RV_myo = (segmentation == 6).astype(np.uint8)
    RV_epi = (RV_endo | RV_myo).astype(np.uint8)

    # convert to contours
    # left ventricle
    contours, hierarchy = cv2.findContours(cv2.inRange(LV_endo, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        LV_endo_pts = np.array([x.tolist() for i,x in enumerate(c[:, 0, :]) if i % 2 == 0], dtype=np.int64)
    else:
        LV_endo_pts = []  

    contours, hierarchy = cv2.findContours(cv2.inRange(LV_epi, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        LV_epi_pts = np.array([x.tolist() for i,x in enumerate(c[:, 0, :]) if i % 2 == 0], dtype=np.int64)
    else:
        LV_epi_pts = []

    # right ventricle
    contours, hierarchy = cv2.findContours(cv2.inRange(RV_endo, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        RV_endo_pts = np.array([x.tolist() for i,x in enumerate(c[:, 0, :]) if i % 2 == 0], dtype=np.int64)
    else:
        RV_endo_pts = []


    contours, hierarchy = cv2.findContours(cv2.inRange(RV_epi, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        RV_epi_pts = []
        for c in contours:
            RV_epi_pts += [x.tolist() for i,x in enumerate(c[:, 0, :]) if i % 2 == 0]
        RV_epi_pts = np.array(RV_epi_pts, dtype=np.int64)
    else:
        RV_epi_pts = []

    contours, hierarchy = cv2.findContours(cv2.inRange(RV_myo, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        RV_myo_pts = []
        for c in contours:
            RV_myo_pts += [x.tolist() for i,x in enumerate(c[:, 0, :]) if i % 2 == 0]
        RV_myo_pts = np.array(RV_myo_pts, dtype=np.int64)

    # la
    contours, hierarchy = cv2.findContours(cv2.inRange(la, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        la_pts = np.array([x.tolist() for i,x in enumerate(c[:, 0, :]) if i % 2 == 0], dtype=np.int64)
    else:
        la_pts = []

    # ra
    contours, hierarchy = cv2.findContours(cv2.inRange(ra, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        ra_pts = np.array([x.tolist() for i,x in enumerate(c[:, 0, :]) if i % 2 == 0], dtype=np.int64)
    else:
        ra_pts = []

    # Get intersection points between RV endo and LV epi to separate septal wall from free wall
    if len(RV_endo_pts)>0 and len(LV_epi_pts)>0:

        pairs = get_intersections(RV_endo_pts, LV_epi_pts, distance_cutoff=1.5)

        if len(pairs) > 0:
            RV_septal_pts = RV_endo_pts[np.unique(pairs[:,0])] # deletes intersection from RV endo pts
            RV_fw_pts = np.array([pnt.tolist() for i, pnt in enumerate(RV_endo_pts) if i not in np.unique(pairs[:,0])], 
                                dtype=np.int64)
            LV_epi_pts = np.array([pnt.tolist() for i, pnt in enumerate(LV_epi_pts) if i not in np.unique(pairs[:,1])], 
                                dtype=np.int64)
            
        else:
            RV_septal_pts = []
            RV_fw_pts = RV_endo_pts
    else:
        RV_septal_pts = []
        RV_fw_pts = RV_endo_pts
            

    # # Get intersection points between RV epi and RV fw to remove extraneous RV epi points
    # if len(RV_epi_pts)>0 and len(RV_fw_pts)>0:
    #     pairs = get_intersections(RV_epi_pts, RV_fw_pts, distance_cutoff=5)

    #     if len(pairs) > 0:
    #         RV_epi_pts = RV_epi_pts[np.unique(pairs[:,0])]

    # Get intersection points between LV endo and la to clean LV endo pts
    if len(LV_endo_pts)>0 and len(la_pts)>0:
        pairs = get_intersections(LV_endo_pts, la_pts, distance_cutoff=1.5)

        if len(pairs) > 0:
            LV_endo_pts = np.array([pnt.tolist() for i, pnt in enumerate(LV_endo_pts) if i not in np.unique(pairs[:,0])], 
                                dtype=np.int64)
            
    # Get intersection points between LV epi and la to clean LV epi pts
    if len(LV_epi_pts)>0 and len(la_pts)>0:
        pairs = get_intersections(LV_epi_pts, la_pts, distance_cutoff=1.5)

        if len(pairs) > 0:
            LV_epi_pts = np.array([pnt.tolist() for i, pnt in enumerate(LV_epi_pts) if i not in np.unique(pairs[:,0])], 
                                dtype=np.int64)
    
    # Get intersection points between RV fw and ra to clean RV fw pts
    if len(RV_fw_pts)>0 and len(ra_pts)>0:
        pairs = get_intersections(RV_fw_pts, ra_pts, distance_cutoff=1.5)

        if len(pairs) > 0:
            RV_fw_pts = np.array([pnt.tolist() for i, pnt in enumerate(RV_fw_pts) if i not in np.unique(pairs[:,0])], 
                                dtype=np.int64)
    
    # Get intersection points between RV epi and RV myo to keep only free wall points
    if len(RV_epi_pts)>0 and len(RV_myo_pts)>0:
        pairs = get_intersections(RV_epi_pts, RV_myo_pts, distance_cutoff=1.5)

        if len(pairs) > 0:
            RV_epi_pts = RV_epi_pts[np.unique(pairs[:,0])]

    
    # Remove intersection between RV epi and LV epi
    if len(RV_epi_pts)>0 and len(LV_epi_pts)>0:
        pairs = get_intersections(RV_epi_pts, LV_epi_pts, distance_cutoff=1.5)

        if len(pairs) > 0:
            RV_epi_pts = np.array([pnt.tolist() for i, pnt in enumerate(RV_epi_pts) if i not in np.unique(pairs[:,0])], 
                                dtype=np.int64)
            
    return [LV_endo_pts, LV_epi_pts, RV_septal_pts, RV_fw_pts, la_pts, ra_pts, RV_epi_pts]

