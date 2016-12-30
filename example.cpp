#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include "jpeglib.h"

int main(int argc, char *argv[]){
  char *filename = (char *)"test_jpeg.jpg";
  int quality    = 50;

  struct jpeg_compress_struct cinfo;    // Basic info for JPEG properties.
  struct jpeg_error_mgr jerr;           // In case of error.
  FILE * outfile;                       // Target file.
  JSAMPROW row_pointer[1];              // Pointer to JSAMPLE row[s].
  int row_stride;                       // Physical row width in image buffer.

  //## ALLOCATE AND INITIALIZE JPEG COMPRESSION OBJECT

  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);

  //## OPEN FILE FOR DATA DESTINATION:

  if ((outfile = fopen(filename, "wb")) == NULL){
    fprintf(stderr, "ERROR: can't open %s\n", filename);
    exit(1);
  }
  jpeg_stdio_dest(&cinfo, outfile);

  //## SET PARAMETERS FOR COMPRESSION:

  cinfo.image_width  = 200;          // |-- Image width and height in  pixels.
  cinfo.image_height = 200;          // |
  cinfo.input_components = 3;       // Number of color components per pixel.
  cinfo.in_color_space = JCS_RGB;   // colorspace of input image as RGB.

  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, quality, TRUE);

  //## CREATE IMAGE BUFFER TO WRITE FROM AND MODIFY THE IMAGE TO LOOK LIKE CHECKERBOARD

  unsigned char *image_buffer = NULL;
  image_buffer = (unsigned char*) malloc(cinfo.image_width*cinfo.image_height*cinfo.num_components);

  for(unsigned int y = 0; y < cinfo.image_height; y++){
    for(unsigned int x = 0; x < cinfo.image_width; x++){
      unsigned int pixelIdx = ((y * cinfo.image_height) + x) * cinfo.input_components;

      if(x % 2 == y % 2){
        image_buffer[pixelIdx + 0] = 255;   // r |-- Set r,g,b components  to
        image_buffer[pixelIdx + 1] = 0;     // g |   make this pixel red
        image_buffer[pixelIdx + 2] = 0;     // b |   (255,0,0).
      }
      else {
        image_buffer[pixelIdx + 0] = 255;   // r |-- Set r,g,b components  to
        image_buffer[pixelIdx + 1] = 255;   // g |   make this pixel white
        image_buffer[pixelIdx + 2] = 255;   // b |   (255,255,255).
      }
    }
  }

  //## START COMPRESSION:

  jpeg_start_compress(&cinfo, TRUE);
  row_stride = cinfo.image_width * 3;       //JSAMPLES per row in image_buffer

  while (cinfo.next_scanline < cinfo.image_height){
    row_pointer[0] = &image_buffer[cinfo.next_scanline * row_stride];
    (void) jpeg_write_scanlines(&cinfo, row_pointer, 1);
  }
        // NOTE: jpeg_write_scanlines expects an array of pointers to scanlines.
        //       Here the arrayis only one element long, but you could pass
        //       more than one scanline at time if that's more convinient.

  //## FINISH COMPRESSION AND CLOSE FILE:

  jpeg_finish_compress(&cinfo);
  fclose(outfile);
  jpeg_destroy_compress(&cinfo);

  printf("SUCCESS\n");

  return 0;
}
