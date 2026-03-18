import os
import sys
import fnmatch
import skimage.io as io
import tifffile
from misc.XMLUtils import XMLUtils
import logging
import glob
from misc.TiffTileLoader import TiffTileLoader
import numpy as np


class ImageTiler(object):

    def __init__(self,name,root_dir):
        self.stage_name = name
        self.root_dir = root_dir
        self.nErrors = 0
        self.config = None

        #init logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        # create a file handler
        log_name = os.path.join(root_dir,'ImageTiler.log')
        handler = logging.FileHandler(log_name)
        handler.setLevel(logging.DEBUG)
        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        # add the handlers to the logger
        self.logger.addHandler(handler)

        #default values
        self.PIX_1MM = 2890  # 1mm= 2890 pixels for 10x RO1
        self.PIX_5MM = 14450  # 5mm = 14450 pixels for 10x RO1
        #self.PIX_1MM = 5814  # 1mm= 5814 pixels for 20x Luke Syn
        #self.PIX_5MM = 29070  # 5mm = 29070 pixels for 20x Luke Syn
        self.MEM_MAX = '14Gb'
        self.SCRIPT_DIR=''


    def get_stage_name(self):
        return self.stage_name

    def set_config(self,config):
        self.config = config
        if self.config:
            self.MEM_MAX = str(self.config.get('global', 'MAGICK_MEM'))
            self.PIX_1MM = int(self.config.get('global', 'PIX_1MM'))
            self.PIX_5MM = int(self.config.get('global', 'PIX_5MM'))
            self.SCRIPT_DIR = str(self.config.get('global', 'SCRIPT_DIR'))

    def run_stage(self):
        # root_dir = '/home/maryana/storage/Posdoc/AVID/AV13/AT100/full_res'
        # root_dir= '/Users/maryana/Posdoc/AVID/AV13/TEMP'
        self.tile_images(self.root_dir)

        return self.nErrors


    def save_metadata(self,img_name,info_dic,log_file):
        tiles = info_dic['tile_grid']

        tile_info = {'name':'Tiles','attrib':{'grid_rows':str(int(tiles[0])),'grid_cols':str(int(tiles[1]))}}
        s = info_dic['size']
        img_info = {'name':'Image', 'attrib':{'rows':str(s[0]), 'cols':str(s[1]), 'file':img_name, 'home':info_dic['home'], 'children':[tile_info]}}
        XMLUtils.dict2xmlfile(img_info,log_file)


    def check_num_tiles(self,tiles_dir,correct_num):
        flist = glob.glob(tiles_dir+'/*.tif')
        if len(flist) != correct_num:
            print('Numbers of tiles are different from calculation:  {} tiles in reality but {} tiles through calculation!'.format(len(flist),correct_num))
            return False
        else:
            print('Numbers of tiles are correct!')
            return True


    def tile_images(self,root_dir):
        self.logger.info('Searching for images to tile.')
        
        # 1. Find all TIFF files to be processed
        file_list = []
        for root, _, files in os.walk(root_dir):
            if fnmatch.fnmatch(root, '*/RES'):
                # Check for both .tif and .tiff extensions
                matches = fnmatch.filter(files, '*.tif') + fnmatch.filter(files, '*.tiff')
                for fn in matches:
                    if 'res10' not in fn:
                        file_list.append(os.path.join(root, fn))

        # 2. Process each file sequentially
        for fi in file_list:
            self.logger.info('*** Processing %s ***', fi)
            print('Processing {}'.format(fi))

            home_dir = os.path.dirname(fi)

            # Check if file was already processed. If so, skip it.
            if os.path.exists(os.path.join(home_dir, 'tiles/tiling_info.xml')):
                self.logger.info('File tiles/tiling_info.xml exists. Skipping this image.')
                print('File {} has already been tiled. Nothing to do.'.format(fi))
                continue

            # 3. Use TiffTileLoader to get reliable dimensions
            tiffLoader = TiffTileLoader(self.PIX_1MM, self.PIX_5MM)
            self.logger.info('Opening full resolution image with TiffTileLoader')
            tiffLoader.open_file(fi)
            size = tiffLoader.get_file_dim()  # [height, width, channels]

            if size[0] == 0 or size[1] == 0:
                self.logger.error('Failed to read dimensions for %s. Skipping.', fi)
                print(f'  - ERROR: Failed to read dimensions for {fi}. Skipping.')
                self.nErrors += 1
                continue

            # 4. Calculate tile grid using the correct dimensions
            # Use np.ceil to ensure at least one tile is created for small images
            nB_rows = np.ceil(size[0] / self.PIX_1MM)
            nB_cols = np.ceil(size[1] / self.PIX_1MM)
            tile_grid = [nB_rows, nB_cols]

            #create tiles directory
            tiles_dir = os.path.join(home_dir, 'tiles')
            if not os.path.exists(tiles_dir):
                self.logger.info('Creating tiles folder %s', tiles_dir)
                os.mkdir(tiles_dir, 0o0777)

            #tile names
            str_tname = 'tile_{:04}.tif'
            str_tname = os.path.join(tiles_dir, str_tname)

            # 5. Compute tile coordinates with the correct grid size
            self.logger.info('Computing tile coordinates')
            tiffLoader.compute_tile_coords(tile_grid[0], tile_grid[1])
            
            # check if coordinates yield same size as original image
            if not tiffLoader.coords_sanity_check(tile_grid[0], tile_grid[1]):
                self.logger.info('Coord sanity: Total tiled images size differs from original image size')
                print('Coord sanity check NOT OK')
            else:
                self.logger.info('Coords sanity check OK')
                print('Coord sanity check OK')
            
            #run tiling
            tile_iterator = tiffLoader.get_tile_iterator()
            count = 0
            self.logger.info('Beginning to save tiles')
            print('Saving tiles...')
            for tile in tile_iterator:
                tile_name = str_tname.format(count)
                io.imsave(tile_name, tile)
                count += 1
            self.logger.info('Finished saving tiles')

            #check if all tiles were saved
            if self.check_num_tiles(tiles_dir, int(tile_grid[1]) * int(tile_grid[0])):
                # Build info dictionary for metadata saving
                fdic = {'home': home_dir, 'size': size, 'tile_grid': tile_grid}
                meta_file = os.path.join(tiles_dir, 'tiling_info.xml')
                self.save_metadata(fi, fdic, meta_file)
                self.logger.info('Metadata saved.')
                coods_file = os.path.join(tiles_dir, 'tile_coordinates.npy')
                np.save(coods_file, tiffLoader.get_tile_coords())
                self.logger.info('Tile coodinates matrix saved.')

            # check if tiles have the same size of the original image
            if not tiffLoader.sanity_check(tiles_dir, tile_grid[0], tile_grid[1]):
                self.logger.info('Total tiled images size differs from original image size')
                print('Sanity check NOT OK')
            else:
                self.logger.info('Sanity check OK')
                print('Sanity check OK')


def main():
    if len(sys.argv) != 2:
        print('Usage: ImageTiler.py <root_dir>')
        exit()

    root_dir = str(sys.argv[1])  # abs path to where the images are
    imgTiler = ImageTiler('Tile Image Cluster',root_dir)
    #root_dir = '/home/maryana/storage/Posdoc/AVID/AV13/AT100/full_res'
    #root_dir= '/Users/maryana/Posdoc/AVID/AV13/TEMP'
    #tile_images(root_dir)
    imgTiler.run_stage()

if __name__ == '__main__':
    main()
