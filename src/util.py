from os import path
import config
import dataprovider

SRC_PATH = path.dirname(path.abspath(__file__))
DATA_PATH = path.join(SRC_PATH, '..', 'data')

YEARBOOK_PATH = path.join(DATA_PATH, "yearbook", "yearbook")

# Get the label for a file
# For yearbook this returns a year
# For streetview this returns a (longitude, latitude) pair
# def label(filename):
  # m = yb_r.search(filename)
  # if m is not None: return int(m.group(1))
  # m = sv_r.search(filename)
  # assert m is not None, "Filename '%s' malformatted"%filename
  # return float(m.group(2)), float(m.group(1))

# List all the yearbook files:
#   train=True, valid=False will only list training files (for training)
#   train=False, valid=True will only list validation files (for testing)
def listIms(train=True, valid=True, test=True):
  r = []
  if train:
      for i in range(len(config.allTrainImNames)):
          for j in range(len(config.allTrainImNames[i])):
              r = r + [(config.allTrainImNames[i][j][0], i)]
  if valid:
      return dataprovider.get_samples(config.valid_data, config.input_shape), config.valid_labels
      # for i in range(len(config.test1ImNames)):
      #     for j in range(len(config.test1ImNames[i])):
      #         r = r + [(config.test1ImNames[i][j][0], i)]
  if test:
      for i in range(len(config.test2ImNames)):
          for j in range(len(config.test2ImNames[i])):
              r = r + [(config.test2ImNames[i][j][0], i)]
  return r
