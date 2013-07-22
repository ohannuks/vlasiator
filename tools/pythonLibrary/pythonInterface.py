# Import files
visitImported = False
import sys
import subprocess
from numpy import *
from termcolor import colored
if visitImported == False:
   from visit import *
   visitImported = True
from useroptions import *
sys.path.insert(0, pythonLibDirectoryPath + pyVlsvPath)
sys.path.insert(0, pythonLibDirectoryPath + pyVisitPath)
sys.path.insert(0, pythonLibDirectoryPath + pyMiscPath)
from vlsvreader import *
from makeamovie import *
from spacecraft import *
from distributionplot import *
from movingframeofreference import *
from movingline import *

visitLaunched = False

def launch_visit(noWindow=True):
   '''
   Launches visit
   :param noWindow=True   Determines whether window is shown or not
   '''
   global visitLaunched
   if visitLaunched == True:
      print "Visit already launched"
      return
   if noWindow == True:
      LaunchNowin(vdir=pathToVisit)
   else:
      Launch(vdir=pathToVisit)
   visitLaunched = True
   print "Visit launched"
   return

def list_functionality():
   print "Class: " + colored("VlsvFile", "red")
   print "   Example usage:"
   print "      myReader = VlsvFile(\"distribution.vlsv\") -- opens a vlsv file for reading"
   print "      myReader.list() -- Gives a list of attributes in the vlsv file"
   print "      myReader.read(name=\"rho\", tag=\"VARIABLE\", mesh=\"SpatialGrid\", read_single_cellid=-1) -- Reads array with the name 'rho', tag 'VARIABLE', mesh 'SpatialGrid'. if read_single_cellid is specified then the reader reads only the given cell id, if specified as -1 it reads the whole array."
   print "      myReader.read_variables(\"rho\") -- Reads values of rho in the form of array"
   print "      myReader.read_variable(\"rho\", 16) -- Reads the 16th cell id's value of rho and returns it"
   print "      myReader.read_blocks(16) -- Reads the raw blocks of cell 16\n"
   print "Function: " + colored("launch_visit(noWindow=True)", "red")
   print "   Example usage:"
   print "      launch_visit(noWindow=False) NOTE: this must be launched before doing anything regarding visit\n"
   print "Function: " + colored("make_movie( variableName, minValue, maxValue, inputDirectory, inputFileName, outputDirectory, outputFileName, colorTable=\"hot_desaturated\", startFrame=-1, endFrame=-1 )", "red")
   print "   Example usage:"
   print "      make_movie(variableName=\"rho\", minValue=1.0e6, maxValue=5.0e6, inputDirectory=\"/home/hannukse/meteo/stornext/field/vlasiator/2D/AAJ/silo_files/\", inputFileName=\"bulk.*.silo\", outputDirectory=\"/home/hannukse/MOVINGFRAME_MOVIES/AAJ_BZ_REMAKE/\", outputFileName=\"RHO_FORESHOCK_\", colorTable=\"hot_desaturated\", startFrame=30, endFrame=120)\n"
   print "Function: " + colored("make_movie_auto(BBOX)", "red")
   print "   Example usage:"
   print "      make_movie_auto(BBOX)\n"
   print "Function: " + colored("make_moving_frame_of_reference_movie( x_begin, x_end, y_begin, y_end, speed_x, speed_y, variable_name, minThreshold, maxThreshold, input_directory, input_file_name, output_directory, output_file_name, color_table=\"hot_desaturated\", start_frame=-1, end_frame=-1, frame_skip_dt=1.0 )", "red")
   print "   Example usage:"
   print "      make_moving_frame_of_reference_movie( x_begin=50e6, x_end=150e6, y_begin=-150e6, y_end=-50e6, speed_x=-500000, speed_y=1000, variable_name=\"rho\", minThreshold=1.0e5, maxThreshold=1.0e6, input_directory=\"/stornext/field/vlasiator/AAJ/distributions/\", input_file_name=\"bulk.*.silo\", output_directory=\"/stornext/field/vlasiator/visualizations/\", output_file_name=\"RHO_MOVIE\", color_table=\"hot_desaturated\", start_frame=0, end_frame=85, frame_skip_dt=2.0 )\n"
   print "Function: " + colored("make_moving_frame_of_reference_line_plot(  point1, point2, velocity, variable_name, input_directory, input_file_name, output_directory, output_file_name, start_frame=-1, end_frame=-1, frame_skip_dt=1.0 )", "red")
   print "   Example usage:"
   print "      make_moving_frame_of_reference_line_plot( point1=[40e6, 140e6, 0], point2=[12e6, 523e6, 0], velocity=[-500000, 0, 0], variable_name=\"rho\", input_directory=\"/stornext/field/vlasiator/2D/AAJ/silo_files/\", input_file_name=\"bulk.*.silo\", output_directory=\"/stornext/field/vlasiator/visualizations/movies/\", output_file_name=\"RHO_MOVIE_\", start_frame=10, end_frame=50, frame_skip_dt=1.5 )\n"
   print "Function: " + colored("make_distribution_movie(cellids, rotated, inputDirectory, outputDirectory, outputFileName, zoom=1.0, viewNormal=[0.488281, 0.382966, -0.784167], minThreshold=1e-18, maxThreshold=1e37)", "red")
   print "   Example usage:"
   print "      make_distribution_movie(cellids=[18302, 19432, 19042], rotated=True, inputDirectory=\"/home/hannukse/meteo/stornext/field/vlasiator/2D/AAJ/silo_files/\", outputDirectory=\"/home/hannukse/MOVIES/\", outputFileName=\"testmovie\", zoom=0.8, viewNormal=[0.488281, 0.382966, -0.784167], minThreshold=1e-17, maxThreshold=1.2e37)"
   print "   Note: viewNormal determines the angle of view (straight from visit)\n"
   print "Function: " + colored("make_moving_spacecraft_plot()", "red")
   print "   Example usage:"
   print "      make_moving_spacecraft_plot(\n)"
   print "Function: " + colored("make_cut_through_plot()", "red")
   print "   Example usage:"
   print "      make_cut_through_plot()\n"
