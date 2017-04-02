"""
Author: GBernardino
Scans the nodule candidates, applies a threshold, searches for nodules that are likely to be the same (but in different slices) and joins them. The result  is written in another csv.
Usage from cli:
    python mergeNodules.py inputCSV outputCSV (debug=[True/False])

Implementation considerations (quick and dirty):

- Only considers the things that are likely to be a nodule (ie, applying threshold). A better approach would include information in between, if it's relatively high.
- I have considered that the nodule continuations are more or less concentric, and that there can be a gap in the middle slices where no nodule is detected
- I haven't check if a nodule could be appended to more than one nodule (there is a check in search_in_open_nodules that can be enabled).
- Not optimized, but should run relatively fast (90s in a Macbook Pro for the whole dataset)

Output writes a  csv with:
-
 - The resulting position (nslice, x, y)  -  only approximate.
 - The maximum score and diameter.
 - The spread along slices (nslicesSpread) 
"""
import pandas as pd
import numpy as np
import time
import os

#
# Main function
#
def merge_nodules_csv(path_csv, path_result,nodule_threshold = 0.7,  pixels_to_mm = np.array([2, 0.7, 0.7]), debug = False):
    """
    reads all the nodules from a csv, and creates another csv with the nodules merged if they are near and in different slices 
    """
    tStart = time.time()
    pd_nodules = pd.read_csv(path_csv)
    pd_nodules_filtered = pd_nodules[pd_nodules['score'] > nodule_threshold]
    nodules_by_patient = pd_nodules_filtered.groupby('patientid')
    print 'finished reading the original nodules'

    number_nodes_concatenated = 0
    df_merged_nodules = pd.DataFrame()
    for nPatient, pId in enumerate(nodules_by_patient.groups.keys()):
        if nPatient % 50 == 0:
            print '%d / %d' % (nPatient, len(nodules_by_patient.groups))
        patient_nodules = nodules_by_patient.get_group(pId).sort_values('nslice')
        closed_nodules = []
        open_nodules = []
        num_nodes = len(patient_nodules)
        for i in xrange(num_nodes):
            nodule = patient_nodules.iloc[i]
            concatenateIdx = search_in_open_nodules(nodule, open_nodules, pixels_to_mm)
            if concatenateIdx == -1:
                open_nodules.append([nodule])
            else:
                number_nodes_concatenated += 1
                if debug:
                    print 'meow', nodule, open_nodules[concatenateIdx][-1]
                    open_nodules[concatenateIdx].append(nodule)
                    print '----------'
            if i % 20 == 0:
                close_nodules(open_nodules, closed_nodules, patient_nodules.iloc[i].nslice)
                
        close_nodules(open_nodules, closed_nodules, 1e6) #1e6 = infinity, to close all nodules
        closed_nodules_df = pd.DataFrame([merge_nodules(l) for l in closed_nodules])
        df_merged_nodules = df_merged_nodules.append(closed_nodules_df)
    tFinal = time.time()
    print 'Finished merging. Total %d nodules, in total %d nodules where merged' % ( len(pd_nodules_filtered), number_nodes_concatenated) 
    print 'Time needed: %f' % (tFinal - tStart)
    df_merged_nodules.to_csv(path_result)
#
# Auxiliary functions
#
same_nodule_likelihood_threeshold = .5

def intersection_sphere(sph1, sph2):
    """
    computes the intersection area
    http://mathworld.wolfram.com/Circle-CircleIntersection.html
    """
    raise (NotImplemented)

def distance2D(a, b, pixels_to_mm = np.array([0.7, 0.7])):
    """
    Gets the distance between two slices 
    """
    d = np.array([a.x - b.x, a.y - b.y])
    return np.dot(d, pixels_to_mm)

def same_nodule_likelihood(nodule1, nodule2,  pixels_to_mm) :
    """
    Returns a high value if noudle1 is the continuation of nodule 2 in another slice.
    I am assuming that the nodules are more or less 
    """
    intersect2D =   2.  *distance2D(nodule1, nodule2, pixels_to_mm[1:3]) / (nodule1.diameter + nodule2.diameter)  #Is > 1 if they do not intersect in 2D, < 1 otherwise, 0 if they have the same center
    z_distance = np.abs(nodule1.nslice - nodule2.nslice) * pixels_to_mm[0]
    
    if z_distance > 8:  #If they are more than 8 mm
        return 0
    elif intersect2D >  0.5: #If they aren't concentring
        return 0
    else:
        return 1

def search_in_open_nodules(a, open_nodules, pixels_to_mm, check_only_one_candidate = False):
    """
    @return : index of the nodule it belongs, -1 if it can not be joined to any current nodule.
    
    searches in the list of open nodules if there is a nodule candidate at distance.
    WARNING!!! supposes that there are not two nodules close enough to get confused.
    """
    if check_only_one_candidate and sum([same_nodule_likelihood(a, o[-1]) > same_nodule_likelihood_threeshold for o in open_nodules]) > 1:
        raise Exception("more than one candidate to nodule merging in patient %s!" % a.patientid)
        
    for i, o in enumerate(open_nodules):
        if same_nodule_likelihood(a, o[-1], pixels_to_mm) > same_nodule_likelihood_threeshold:
            return i

    return -1

def close_nodules(open_nodules, closed_nodules, currentSlice, nSlicesClose = 5):
    """
    Closes the nodules that are still open far away from the current slice.
    """
    i = 0
    while i < len(open_nodules):
        if currentSlice - open_nodules[i][-1].nslice >  nSlicesClose:
            closed_nodules.append(open_nodules[i])
            del open_nodules[i]
        else:
            i += 1
def merge_nodules(list_closed_nodules):
    """
    merges nodule to obtain a single row for each.
    The (x, y,z) position is only approximate (the mean of all). Anyway, only the approximate position is relevant, no in a milimiter specific way.
    """
    data = {}
    data['patientid'] = list_closed_nodules[0].patientid
    data['nslice'] = (list_closed_nodules[0].nslice + list_closed_nodules[-1].nslice)/2
    data['x'] =  np.mean(map(lambda n: n.x , list_closed_nodules))
    data['y'] =  np.mean(map(lambda n: n.y , list_closed_nodules))
    data['diameter'] = max(map(lambda n: n.diameter , list_closed_nodules))
    data['score'] = max(map(lambda n: n.score , list_closed_nodules))
    data['nslicesSpread'] = (- list_closed_nodules[0].nslice + list_closed_nodules[-1].nslice) + 1
    return pd.Series(data)

if __name__ == '__main__':
    if len(sys.argv) not in [3, 4]:
        print 'Incorrect arguments. Correct use:'
        print 'python mergeNodules.py inputCSV outputCSV (debug=[True/False])'
    else:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        try:
            debug = sys.argv[3] == 'debug=True'
        except:
            debug = False
        joinNodules.merge_nodules_csv(input_path, output_path, debug = debug)
