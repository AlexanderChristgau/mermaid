"""
This package implements a simple way of dealing with parameters, ofproviding
default parameters and comments, and to keep track of used parameters for 
registration runs. See the corresponding note for a brief description on how to use it.
"""
from __future__ import print_function

from builtins import str
from builtins import object
import json

class ParameterDict(object):
    def __init__(self,initDict=None,printSettings=True):
        if initDict is not None:
            if type(initDict)==type(self):
                self.ext = initDict.ext
            else:
                print('WARNING: Cannot initialize from non ParameterDict object. Ignoring initialization.')
                self.ext = {}
        else:
            self.ext = {}
        self.int = {}
        self.com = {}
        self.currentCategoryName = 'root'
        self.printSettings = printSettings

    def __str__(self):

        return 'ext = ' + self.ext.__str__() + "\n" + \
            'int = ' + self.int.__str__() + "\n" + \
            'com = ' + self.com.__str__() + "\n" + \
            'currentCategoryName = ' + str( self.currentCategoryName) +"\n"

    def isempty(self):
        return self.int=={}

    def load_JSON(self, fileName):
        """
        Loads a JSON configuration file
        
        :param fileName: filename of the configuration to be loaded 
        """
        try:
            with open(fileName) as data_file:
                if self.printSettings:
                    print('Loading parameter file = ' + fileName )
                self.ext = json.load(data_file)
        except IOError as e:
            print('Could not open file = ' + fileName + '; ignoring request.')

    def write_JSON_and_JSON_comments(self,fileNames):
        """
        Writes the JSON configuration to a file

        :param fileNames: filename tuple; first entry is filename for configuration, second for comments
        """
        self.write_JSON(fileNames[0])
        self.write_JSON_comments(fileNames[1])

    def write_JSON(self, fileName):
        """
        Writes the JSON configuration to a file
        
        :param fileName: filename to write the configuration to  
        """

        with open(fileName, 'w') as outfile:
            if self.printSettings:
                print('Writing parameter file = ' + fileName )
            json.dump(self.int, outfile, indent=4, sort_keys=True)

    def write_JSON_comments(self, fileNameComments):
        """
        Writes the JSON commments file. This file will not contain any actual values, but
        instead descriptions of the settings (if they have been provided). The goal is to
        provide self-documenting configuration files.
        
        :param fileNameComments: filename to write the commented JSON configuration to  
        """

        with open(fileNameComments, 'w') as outfile:
            if self.printSettings:
                print('Writing parameter file = ' + fileNameComments )
            json.dump(self.com, outfile, indent=4, sort_keys=True)

    def print_settings_on(self):
        """
        Enable screen output as configurations are read and set.
        """
        self.printSettings = True

    def print_settings_off(self):
        """
        Disable screen output as configurations are read and set.
        """
        self.printSettings = False

    def get_print_settings(self):
        """
        Current print settings
        :return: Returns if screen printing is on (True) or off (False)
        """
        return self.printSettings

    def _set_value_of_instance(self, ext, int, com, currentCategoryName):
        self.ext = ext
        self.int = int
        self.com = com
        self.currentCategoryName = currentCategoryName

    def __missing__(self, key):
        # if key cannot be found
        raise ValueError('Could not find key = ' + str( key ) )

    def __getitem__(self, key_or_keyTuple):
        # getting an item based on key
        # here the key can be three different things
        # 1) simply a text key (then returns the current value)
        # 2) A 2-tuple (keyname,defaultvalue)
        # 3) A 3-tuple (keyname,defaultvalue,comment)

        # returns a ParDicts object if we are accessing a category (i.e., a dictionary)
        # returns just the value if it is a regular value

        if type(key_or_keyTuple)==tuple:
            # here, we need to distinguish cases 2) and 3)
            lT = len(key_or_keyTuple)
            if lT==1:
                # treat this as if it would only be the keyword
                return self._get_current_key(key_or_keyTuple[0])
            elif lT==2:
                # treat this as keyword + default value
                return self._get_current_key(key_or_keyTuple[0],
                                             key_or_keyTuple[1])
            elif lT==3:
                # treat this as keyword + default value + comment
                return self._get_current_key(key_or_keyTuple[0],
                                             key_or_keyTuple[1],
                                             key_or_keyTuple[2])
            else:
                raise ValueError('Tuple of incorrect size')
        else:
            # now we just want to return it (there is no default value or comment)
            return self._get_current_key(key_or_keyTuple)


    def __setitem__(self, key, valueTuple):
        # to set an item
        # valueTuple is either a 2-tuple (actual value, comment)
        # or it is simply a comment, then this key becomes a category
        if type(valueTuple)==tuple:
            if len(valueTuple)==2:
                value = valueTuple[0]
                comment = valueTuple[1]
            elif len(valueTuple)==1:
                value = {}
                comment = valueTuple[0]
            else:
                raise ValueError('Expected a 2-tuple as input')
        else: # not a tuple
            value = valueTuple
            comment = None

        if type(value)==dict:
            # only add if this is an empty dictionary
            if len(value)==0:
                self._set_current_category(key, comment)
            else:
                raise ValueError('Can only add empty dictionaries')
            # we are assigning a category
        else:
            # now we have to set an actual value (not a category)
            if type(value)==type(self):
                # Here we are trying to assign a full parameter object
                # We want to add the content and not the object itself
                self.ext[key]=value.ext
                self.int[key]={}
                self.com[key]={}
            else:
                # this is just a normal value
                self._set_current_key(key, value, comment)

    def _set_current_category(self, key, comment):
        currentCategoryName = self.currentCategoryName + '.' + str(key)

        if key not in self.ext or (key in self.ext and type(self.ext[key])!=dict):
            # we do not want to over-write any settings here
            if self.printSettings:
                print('Creating new category: ' + currentCategoryName)
            self.ext[key] = {}

        self.int[key] = {}
        self.com[key] = {}

        if comment is not None:
            if len(comment) > 0:
                self.com[key]['__doc__'] = comment

    def _set_current_key(self, key, value, comment=None):

        if self.printSettings:
            if key in self.ext:
                print('Overwriting key = ' + str(key) + '; category = ' + self.currentCategoryName + '; value =  ' +
                      str( self.ext[key] ) + ' -> ' + str(value) )
            else:
                print('Creating key = ' + str(key) + '; category = ' + self.currentCategoryName + '; value = ' + str(value))

        self.ext[key] = value
        self.int[key] = value
        if comment is not None:
            if len(comment)>0:
                self.com[key] = comment


    def _get_current_key(self, key, defaultValue=None, comment=None):

        # returns a ParDicts object if we are accessing a category (i.e., a dictionary)
        # returns just the value if it is a regular value

        if key in self.ext:
            value = self.ext[key]
            if type(value)==dict:
                # this is a category, need to create a ParDicts object to return
                # if the key already exists in int and com keep it otherwise initialize it to empty
                if key not in self.int:
                    self.int[key]={}
                if key not in self.com:
                    self.com[key]={}
                    if comment is not None:
                        if len(comment)>0:
                            self.com[key]['__doc__'] = comment

                newpar = ParameterDict(printSettings=self.printSettings)
                currentCategoryName = self.currentCategoryName + '.' + str(key)
                newpar._set_value_of_instance(self.ext[key], self.int[key], self.com[key], currentCategoryName)

                return newpar
            else:
                # just a regular value which we can return
                self.int[key] = value
                if comment is not None:
                    if len(comment)>0:
                        self.com[key] = comment

                return value
        else:
            # does not have the key, create it via the default value
            if defaultValue is None:
                # then make it a dictionary
                defaultValue = {}
            # if defaultValue is not None:
            if type(defaultValue)==dict:
                # make sure it is empty and if it is create a category
                if len(defaultValue)==0:
                    self._set_current_category(key, comment)
                    # and now we need to return it
                    newpar = ParameterDict(printSettings=self.printSettings)
                    currentCategoryName = self.currentCategoryName + '.' + str(key)
                    newpar._set_value_of_instance(self.ext[key], self.int[key], self.com[key], currentCategoryName)

                    return newpar
                else:
                    raise ValueError('Cannot create a default key of type dict()')
            else:
                # now we can create it and return it
                self.ext[key]=defaultValue
                self.int[key]=defaultValue
                if comment is not None:
                    if len(comment)>0:
                        self.com[key]=comment
                if self.printSettings:
                    print('Using default value = ' + str(defaultValue) + ' for key = ' + str(key) + ' of category = ' + self.currentCategoryName  )

                return defaultValue
            #else:
            #    raise ValueError('Cannot create key = ' + str(key) + ' without a default value')


# test it
def test_parameter_dict():
    """
    Convenience testing script (to be converted to an actual test)
    """

    p = ParameterDict()

    # we can directly assign
    p['registration_model'] = ({},'general settings for registration models')
    p['registration_model']['similarity_measure'] = ({},'settings for the similarity measures')
    p['registration_model']['similarity_measure']['type']=('ssd','similarity measure type')
    # we can also ask for a parameter and use a default parameter if it does not exist
    p['registration_model'][('nrOfIterations',10,'number of iterations')]

    # we can also create a new category with default values if it does not exist yet
    p[('new_category',{},'this is a new category')]
    p[('registration_model',{},'this category already existed')]

    # and we can print everything of course
    print(p)

    # lastly we can write it all out as json
    p.write_JSON('test_pars.json')
    p.write_JSON_comments('test_pars_comments.json')



