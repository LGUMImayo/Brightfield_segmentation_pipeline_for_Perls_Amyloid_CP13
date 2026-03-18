import os
import subprocess
import sys
import fnmatch
import skimage.io as io
import tifffile
from misc.XMLUtils import XMLUtils
import logging
import glob
from misc.TiffTileLoader import TiffTileLoader
import numpy as np

#
# This script is supposed to run on the cluster
# Script for automatically tiling the full resolution histology images, using Image Magick


class MaskTiler(object):

    def __init__(self,name,root_dir):
        self.stage_name = name
        self.root_dir = root_dir
        self.nError = 0
        self.config = None

        # init logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        # create a file handler
        log_name = os.path.join(root_dir, 'MaskTiler.log')
        handler = logging.FileHandler(log_name)
        handler.setLevel(logging.DEBUG)
        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        # add the handlers to the logger
        self.logger.addHandler(handler)

        #Default values
        self.PIX_1MM = 2890  # 1mm= 2890 pixels for 10x RO1
        self.PIX_5MM = 14450  # 5mm = 14450 pixels for 10x RO1
        #self.PIX_1MM = 5814  # 1mm= 5814 pixels for 20x Luke Syn
        #self.PIX_5MM = 29070  # 5mm = 29070 pixels for 20x Luke Syn
        self.MAX_MEM = '256Gb'
        self.TMP_DIR = '/tmp'
        self.EROSION_PIXELS = 0  # Full-res erosion pixels (will be scaled for low-res mask)
        self.MASK_SCALE = 10  # res10 mask is 1/10 of full resolution

    def get_stage_name(self):
        return self.stage_name

    def set_config(self,config):
        self.config = config
        if self.config:
            self.MAX_MEM = str(self.config.get('global', 'MAGICK_MEM'))
            self.PIX_1MM = int(self.config.get('global', 'PIX_1MM'))
            self.PIX_5MM = int(self.config.get('global', 'PIX_5MM'))
            if self.config.has_option('global', 'MAGICK_TMPDIR'):
                self.TMP_DIR = str(self.config.get('global', 'MAGICK_TMPDIR'))
            # Optional erosion setting - erode tissue edges to reduce false positives
            if self.config.has_option('tile_masking', 'EROSION_PIXELS'):
                self.EROSION_PIXELS = int(self.config.get('tile_masking', 'EROSION_PIXELS'))
            if self.config.has_option('tile_masking', 'MASK_SCALE'):
                self.MASK_SCALE = int(self.config.get('tile_masking', 'MASK_SCALE'))


    def run_stage(self):
        # root_dir = '/home/maryana/storage/Posdoc/AVID/AV13/AT100/full_res'
        # root_dir= '/Users/maryana/Posdoc/AVID/AV13/TEMP'
        self.process_masks(self.root_dir)

        return self.nError


    def get_mask_info(self,root_dir):
        file_list = dict()
        mask_list = dict()

        #get info from all histology images
        for root, dir, files in os.walk(root_dir):
            if fnmatch.fnmatch(root,'*/RES'): #it's inside /RES*
                for fn in fnmatch.filter(files,'*.tiff'): #get only full resolution images
                    if fn.find('res10') > -1: #skip res10 images
                        continue
                    
                    file_name = os.path.join(root,fn)
                    
                    # Use the robust TiffTileLoader to get dimensions
                    tiff = TiffTileLoader(self.PIX_1MM, self.PIX_5MM)
                    tiff.open_file(file_name)
                    size = tiff.get_file_dim() # [height, width, channels]

                    if size[0] == 0 or size[1] == 0:
                        self.logger.error('Failed to read dimensions for %s. Skipping.', file_name)
                        print(f'  - ERROR: Failed to read dimensions for {file_name}. Skipping.')
                        continue

                    #try to load tile grid from the histology tiling process
                    tiles_dir = os.path.join(root,'tiles') # Correctly locate tiles dir relative to image
                    meta_xml = os.path.join(tiles_dir, 'tiling_info.xml')
                    if os.path.exists(tiles_dir) and os.path.exists(meta_xml):
                        grid_rows, grid_cols, img_rows, img_cols, img_home, img_file = XMLUtils.parse_tiles_metadata(meta_xml)
                        tile_grid = [grid_rows,grid_cols]

                        if size[0] != img_rows or size[1] != img_cols:
                            self.logger.info('WARNING: image size in tiling_info.xml differs from size read from image file.({}x{}/{}x{})'.format(size[0],size[1],img_rows,img_cols))
                            print('WARNING: image size in tiling_info.xml differs from size read from image file.({}x{}/{}x{})'.format(size[0],size[1],img_rows,img_cols))

                    else:
                        # compute tile grid from file_size using np.ceil to avoid zero grid
                        nB_rows = np.ceil(size[0] / self.PIX_1MM)
                        nB_cols = np.ceil(size[1] / self.PIX_1MM)
                        tile_grid = [nB_rows, nB_cols]
                    file_list[file_name] = {'home':root, 'size':size, 'tile_grid':tile_grid}
                    del tiff

        #find respective masks, files without masks are ignored
        for f in file_list.keys():
            #get masks folder
            home_dir = file_list[f]['home']
            path_list = home_dir.split(os.sep)
            nW = len(path_list)
            mask_dir = ''
            for d in range(nW-2):
                mask_dir+=path_list[d]+os.sep
            mask_dir = os.path.join(mask_dir,'mask/final_mask')
            if os.path.exists(mask_dir):
                mask_file = ''
                mask_root = ''
                for root, dir, files in os.walk(mask_dir):
                    # Check for both .tif and .tiff extensions
                    matches = fnmatch.filter(files, '*mask.tif') + fnmatch.filter(files, '*mask.tiff')
                    for fm in matches:  # there should be only one
                        if fm.find('brain') != -1 or fm.find('wm') != -1:
                            continue
                        mask_root = root
                        mask_file = os.path.join(mask_root,fm)

                if mask_file:
                    mask_list[mask_file] = {'home':mask_root, 'full_size':file_list[f]['size'], 'tile_grid':file_list[f]['tile_grid']}

        return mask_list


    def save_metadata(self,img_name,info_dic,log_file):

        tiles = info_dic['tile_grid']
        tile_info = {'name':'Tiles','attrib':{'grid_rows':str(int(tiles[0])),'grid_cols':str(int(tiles[1]))}}
        s = info_dic['full_size']
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


    def process_masks(self,root_dir):

        #create Image Magick tmp directory
        magick_tmp_dir = os.path.join(self.TMP_DIR, 'magick_tmp')
        if not os.path.exists(magick_tmp_dir):
            try:
                os.makedirs(magick_tmp_dir, 0o777)
            except OSError as e:
                self.logger.error("Failed to create temp directory %s: %s", magick_tmp_dir, e)
                print(f"Error: Could not create temp directory {magick_tmp_dir}. Check permissions and if the parent directory exists.")
                # Fallback to a directory inside root_dir
                magick_tmp_dir = os.path.join(root_dir, 'magick_tmp')
                if not os.path.exists(magick_tmp_dir):
                    os.makedirs(magick_tmp_dir, 0o777)
                self.logger.info("Falling back to temp directory: %s", magick_tmp_dir)
                print(f"Falling back to temp directory: {magick_tmp_dir}")

        #export Image Magick env variables
        os.environ['MAGICK_TMPDIR'] = magick_tmp_dir
        os.environ['MAGICK_MEMORY_LIMIT'] = self.MAX_MEM

        #get file information and tiling grid size

        self.logger.info('Readind files info')
        file_dic = self.get_mask_info(root_dir)
        self.logger.debug('File info %s',file_dic)


        for fi in file_dic.keys():

            self.logger.info('*** Processing %s ***',fi)

            fdic = file_dic[fi]
            home_dir = fdic['home']
            full_size = fdic['full_size']
            tile_grid = fdic['tile_grid']

            # Check if file was already processed. If so, skip it.
            if os.path.exists(os.path.join(home_dir, 'mask_tiling_info.xml')):
                self.logger.info('File mask_tiling_info.xml exists. Skipping this file.')
                print('File {} has already been tiled. Nothing to do.'.format(fi))
                continue

            #create tiles directory
            tiles_dir = os.path.join(home_dir,'tiles')
            if not os.path.exists(tiles_dir):
                self.logger.info('Creating tiles folder.')
                os.mkdir(tiles_dir, 0o777)

            ####
            # Erosion (optional) - erode tissue edges before resizing
            ####
            input_mask_file = fi  # Default: use original mask file
            eroded_temp_file = None
            
            if self.EROSION_PIXELS > 0:
                self.logger.info('Applying erosion to mask before resizing.')
                print(f"Applying {self.EROSION_PIXELS}px erosion to mask (scaled to {self.EROSION_PIXELS // self.MASK_SCALE}px at res{self.MASK_SCALE})")
                
                # Load the low-res mask
                mask_img = io.imread(fi)
                if mask_img.ndim > 2:
                    mask_img = mask_img[:, :, 0]
                
                # Scale erosion pixels for low-res mask
                scaled_erosion = max(1, self.EROSION_PIXELS // self.MASK_SCALE)
                
                # Create circular erosion kernel (disk)
                from scipy import ndimage
                y, x = np.ogrid[-scaled_erosion:scaled_erosion+1, -scaled_erosion:scaled_erosion+1]
                kernel = x**2 + y**2 <= scaled_erosion**2
                
                # Apply erosion
                binary_mask = mask_img > 0
                eroded_mask = ndimage.binary_erosion(binary_mask, structure=kernel)
                
                # Create eroded mask with original values
                mask_eroded = np.zeros_like(mask_img)
                mask_eroded[eroded_mask] = mask_img[eroded_mask]
                
                # Save to temp file
                eroded_temp_file = os.path.join(home_dir, 'temp_eroded_mask.tif')
                tifffile.imwrite(eroded_temp_file, mask_eroded)
                input_mask_file = eroded_temp_file
                
                self.logger.info(f'Erosion applied. Using temp file: {eroded_temp_file}')
                print(f"Erosion complete. Eroded {np.sum(binary_mask) - np.sum(eroded_mask)} pixels from edges.")

            ####
            # Resizing
            ####
            self.logger.info('Beginning resizing.')

            log_out_name1 = os.path.join(home_dir,'resize_stdout_log.txt')
            log_err_name1 = os.path.join(home_dir,'resize_stderr_log.txt')
            # log_out1 = open(log_out_name1,'wb+')
            # log_err1 = open(log_err_name1,'wb+')
            size_str = '{}x{}!'.format(full_size[1],full_size[0]) #image magick uses cols x rows convention

            filename = os.path.splitext(os.path.basename(fi))[0]
            str_name = os.path.join(home_dir,filename+'_full_res.tif')
            str_rname = 'TIFF64:' + str_name #save as BigTiff
            self.logger.info('Resized file: %s', str_name)

            # run resize system process
            print("Resizing file {}".format(input_mask_file))
            # status = subprocess.call(['convert', '-resize', size_str, fi, str_rname], env=dict(os.environ), stderr=log_err1, stdout=log_out1)
            
            command = ['magick', 'convert', '-monitor', '-resize', size_str, input_mask_file, str_rname]
            self.logger.info('Executing command: %s', ' '.join(command))
            process = subprocess.Popen(command, env=dict(os.environ), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

            with process.stdout:
                for line in iter(process.stdout.readline, ''):
                    sys.stdout.write(line)
                    self.logger.info(line.strip())
            
            status = process.wait()


            self.logger.info('Resizing ended. Status: %s',str(status))

            # Clean up temp eroded mask file
            if eroded_temp_file and os.path.exists(eroded_temp_file):
                os.remove(eroded_temp_file)
                self.logger.info(f'Removed temp eroded mask file: {eroded_temp_file}')

            if status == 0:
                ####
                # Tiling
                ####

                #check if resized mask size is correct
                tiff = tifffile.TiffFile(str_name)  # load tiff header only
                mask_size = tiff.series[0].shape
                if mask_size[0] != full_size[0] or mask_size[1] != full_size[1]:
                    self.logger.info('Size of rescaled mask differs from original image size. Skipping this image.')
                    continue

                self.logger.info('Beginning tiling.')
                # tile names
                str_tname = 'tile_{:04}.tif'
                str_tname = os.path.join(tiles_dir, str_tname)

                tiffLoader = TiffTileLoader(self.PIX_1MM, self.PIX_5MM)
                #load resized mask file
                self.logger.info('Opening full resolution image')
                tiffLoader.open_file(str_name)
                self.logger.info('Computing tile coordinates')
                # compute tiles coordinates
                tiffLoader.compute_tile_coords(tile_grid[0], tile_grid[1])

                # check if coordinates yield same size as original image
                if not tiffLoader.coords_sanity_check(tile_grid[0], tile_grid[1]):
                    self.logger.info('Coord sanity: Total tiled images size differs from original image size')
                    print('Coord sanity check NOT OK')
                else:
                    self.logger.info('Coords sanity check OK')
                    print('Coord sanity check OK')

                # run tiling
                tile_iterator = tiffLoader.get_tile_iterator()  # the iterator makes sure the tiles are always in the right order
                count = 0
                self.logger.info('Beginning to save tiles')
                print('Saving tiles...')
                for tile in tile_iterator:
                    tile_name = str_tname.format(count)
                    io.imsave(tile_name, tile)
                    count += 1
                self.logger.info('Finished saving tiles')

                # check if all tiles were saved, save metadata and coodinates file
                if self.check_num_tiles(tiles_dir, tile_grid[1] * tile_grid[0]):
                    # save metadata (used by export_heatmap_metadata.py)
                    meta_file = os.path.join(tiles_dir, 'tiling_info.xml')
                    self.save_metadata(fi, fdic, meta_file)  # save metadata
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

            else:
                self.logger.info('Resizing failed. Cannot continue.')
                self.nError += 1


def main():
    if len(sys.argv) != 2:
        print('Usage: ImageTiler.py <root_dir>')
        exit()

    root_dir = str(sys.argv[1])
    maskTiler = MaskTiler('Mask Tiler',root_dir)
    maskTiler.process_masks(root_dir)

if __name__ == '__main__':
    main()
