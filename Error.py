# define Python user-defined exceptions
import sys

class Error(Exception):
   """Base class for other exceptions"""
   pass

class NoEDPointsError(Error):
   '''There is no ED points for this case'''
   
class NoESPointsError(Error):
   '''There is no ES points for this case'''

class SliceShiftFailed(Error):
   '''The breath-hold misregistration correction has failed'''
