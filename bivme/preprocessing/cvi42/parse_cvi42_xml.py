# Copyright 2017, Wenjia Bai. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
    Parser for cvi42 exported xml files.

    This parser searches for dicom UIDs in the xml file, extract the contour
    point coordinates and save them in a pickle file for each image slice.
    """
import os
import sys
import pickle
import numpy as np
from xml.dom import minidom


def keepElementNodes(nodes):
    """ Get the element nodes """
    nodes2 = []
    for node in nodes:
        if node.nodeType == node.ELEMENT_NODE:
            nodes2 += [node]
    return nodes2


def parseContours(node):
    """
        Parse a Contours object. Each Contours object may contain several contours.
        We first parse the contour name, then parse the points and pixel size.
        """
    contours = {}
    for child in keepElementNodes(node.childNodes):
        contour_name = child.getAttribute('Hash:key')
        sup = 1
        for child2 in keepElementNodes(child.childNodes):
            if child2.getAttribute('Hash:key') == 'Points':
                points = []
                for child3 in keepElementNodes(child2.childNodes):
                    x = float(child3.getElementsByTagName('Point:x')[0].firstChild.data)
                    y = float(child3.getElementsByTagName('Point:y')[0].firstChild.data)
                    points += [[x, y]]
            if child2.getAttribute('Hash:key') == 'SubpixelResolution':
                sub = int(child2.firstChild.data)
        points = np.array(points)
        points /= sub
        contours[contour_name] = points
    return contours


def traverseNode(node, uid_contours):
    """ Traverse the nodes """
    child = node.firstChild
    while child:
        if child.nodeType == child.ELEMENT_NODE:
            # This is where the information for each dicom file starts
            if child.getAttribute('Hash:key') == 'ImageStates':
                for child2 in keepElementNodes(child.childNodes):
                    # UID for the dicom file
                    uid = child2.getAttribute('Hash:key')
                    for child3 in keepElementNodes(child2.childNodes):
                        if child3.getAttribute('Hash:key') == 'Contours':
                            contours = parseContours(child3)
                            if contours:
                                uid_contours[uid] = contours
        traverseNode(child, uid_contours)
        child = child.nextSibling


def parseFile(xml_name, output_dir):
    """ Parse a cvi42 xml file """
    dom = minidom.parse(xml_name)
    uid_contours = {}
    traverseNode(dom, uid_contours)

    # Save the contours for each dicom file
    for uid, contours in uid_contours.items():
        with open(os.path.join(output_dir, '{0}.pickle'.format(uid)), 'wb') as f:
            pickle.dump(contours, f)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: {0} cvi_xml output_dir'.format(sys.argv[0]))
        exit(0)

    parseFile(sys.argv[1], sys.argv[2])