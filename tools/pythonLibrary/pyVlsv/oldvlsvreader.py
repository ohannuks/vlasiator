import struct
import xml.etree.ElementTree as ET
import ast
import numpy as np
   

class VlsvFile(object):
   ''' Class for reading VLSV files
   ''' 
   def __init__(self, file_name):
      self.__file_name = file_name
      self.__xml_root = ET.fromstring("<VLSV></VLSV>")
      self.__fileindex_for_cellid={}
      self.__read_xml_footer()
      self.__read_fileindex_for_cellid()

   def __read_xml_footer(self):
      ''' Reads in the XML footer of the VLSV file and store all the content
      ''' 
      max_xml_size = 1000000
      fptr = open(self.__file_name,"rb")
      #(endianness,) = struct.unpack("c", fptr.read(1))
      fptr.seek(8)
      (offset,) = struct.unpack("Q", fptr.read(8))
      fptr.seek(offset)
      xml_data = fptr.read(max_xml_size)
      fptr.close() 
      (xml_string,) = struct.unpack("%ds" % len(xml_data), xml_data)
      self.__xml_root = ET.fromstring(xml_string)

   def __read_fileindex_for_cellid(self):
      """ Read in the cell ids and create an internal dictionary to give the index of an arbitrary cellID
      """
      cellids=self.read(name="SpatialGrid",tag="MESH")
      for index,cellid in enumerate(cellids):
         self.__fileindex_for_cellid[cellid]=index
      

   def list(self):
      ''' Print out a description of the content of the file. Useful
         for interactive usage
      '''
      print "tag = PARAMETERS"
      for child in self.__xml_root:
         if child.tag == "PARAMETERS":
            print "   ", child.attrib["name"], " = ", child.attrib["value"]
      print "tag = VARIABLE"
      for child in self.__xml_root:
         if child.tag == "VARIABLE":
            print "   ", child.attrib["name"]
      print "Other:"
      for child in self.__xml_root:
         if child.tag != "PARAMETERS" and child.tag != "VARIABLE":
            print "    tag = ", child.tag, " name = ", child.attrib["name"]

   def read(self, name="", tag="", mesh="", read_single_cellid=-1):
      ''' Read data from the open vlsv file. 
      
      Arguments:
      :param name Name of the data array
      :param tag  Tag of the data array. Defaults to VARIABLE.
      :param read_single_cellid  If -1 then all data is read. If nonzero then only the vector for the specified cell id is read
      :returns numpy array with the data

      '''
      if tag == "" and name == "" and tag == "":
         print "Bad arguments at read"
      #TODO, read_single_cellid should perhaps be an list/numpy array with cellids that are read in. This could be more efficient to 
      #     study multiple cells, e.g., along a line
      for child in self.__xml_root:
         if tag != "":
            if child.tag != tag:
               continue
         if name != "":
            if child.attrib["name"] != name:
               continue
         if mesh != "":
            if child.attrib["mesh"] != mesh:
               continue
         if child.tag == tag and child.attrib["name"] == name:
            vector_size = ast.literal_eval(child.attrib["vectorsize"])
            array_size = ast.literal_eval(child.attrib["arraysize"])
            element_size = ast.literal_eval(child.attrib["datasize"])
            datatype = child.attrib["datatype"]            
            offset = ast.literal_eval(child.text)
            if read_single_cellid >= 0:
               offset=offset+self.__fileindex_for_cellid[read_single_cellid]*element_size*vector_size
               array_size=1

            fptr = open(self.__file_name, "rb")
            fptr.seek(offset)

            if datatype == "float" and element_size == 4:
               data = np.fromfile(fptr, dtype = np.float32, count=vector_size*array_size)
            if datatype == "float" and element_size == 8:
               data = np.fromfile(fptr, dtype=np.float64, count=vector_size*array_size)
            if datatype == "int" and element_size == 4:
               data = np.fromfile(fptr, dtype=np.int32, count=vector_size*array_size)
            if datatype == "int" and element_size == 8:
               data = np.fromfile(fptr, dtype=np.int64, count=vector_size*array_size)
            if datatype == "uint" and element_size == 4:
               data = np.fromfile(fptr, dtype=np.uint32, count=vector_size*array_size)
            if datatype == "uint" and element_size == 8:
               data = np.fromfile(fptr, dtype=np.uint64, count=vector_size*array_size)
            fptr.close() 

            if vector_size > 1:
               data=data.reshape(array_size, vector_size)
            
            if array_size == 1:
               return data[0]
            else:
               return data

   def read_variables(self, _name):
      ''' Read variables from the open vlsv file. 
      
      Arguments:
      :param _name Name of the variable
      :returns numpy array with the data

      '''
      return self.read(mesh="SpatialGrid", name=_name, tag="VARIABLE", read_single_cellid=-1)

   def read_variable(self, _name, cell_id):
      ''' Read a variable of a given cell from the open vlsv file. 
      
      Arguments:
      :param _name Name of the variable
      :param _CellID Cell's cell id
      :returns numpy array with the data

      '''
      return self.read(mesh="SpatialGrid", name=_name, tag="VARIABLE", read_single_cellid=cell_id)

   def read_blocks(self, cell_id):
      ''' Read raw block data from the open file. Not so useful yet as 
         there is no information on the location of each block.
      
      Arguments:
      :param cell_id Cell ID of the cell whose velocity blocks are read
      :returns numpy array with blocks in the cell. Empty if cell has no stored blocks.
      '''
         
      #these two arrays are in the same order: 
      #list of cells for which dist function is saved
      cells_with_blocks = self.read("SpatialGrid","CELLSWITHBLOCKS")
      #number of blocks in each cell for which data is stored
      blocks_per_cell = self.read("SpatialGrid","BLOCKSPERCELL")
      (cells_with_blocks_index,) = np.where(cells_with_blocks == cell_id)

      if len(cells_with_blocks_index) == 0:
         #block data did not exist
         return []

      num_of_blocks = blocks_per_cell[cells_with_blocks_index[0]]

      for child in self.__xml_root:
         if child.attrib["name"] == "avgs":
            vector_size = ast.literal_eval(child.attrib["vectorsize"])
            #array_size = ast.literal_eval(child.attrib["arraysize"])
            element_size = ast.literal_eval(child.attrib["datasize"])
            datatype = child.attrib["datatype"]
            
            offset = ast.literal_eval(child.text)
            
            for i in range(0, cells_with_blocks_index[0]):
               offset += blocks_per_cell[i]*vector_size*element_size

            fptr = open(self.__file_name,"rb")
            fptr.seek(offset)
            if datatype == "float" and element_size == 4:
               data = np.fromfile(fptr, dtype = np.float32, count = vector_size*num_of_blocks)
            if datatype == "float" and element_size == 8:
               data = np.fromfile(fptr, dtype = np.float64, count = vector_size*num_of_blocks)
            fptr.close()
            return data.reshape(num_of_blocks, vector_size)   

   
      return []
         
