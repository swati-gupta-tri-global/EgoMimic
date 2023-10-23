#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

#include "LeapC.h"
#include "ExampleConnection.h"
#include <librealsense2/rs.h>
#include <librealsense2/h/rs_pipeline.h>
#include <librealsense2/h/rs_frame.h>

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "example.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                     These parameters are reconfigurable                                        //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define STREAM RS2_STREAM_COLOR // rs2_stream is a types of data provided by RealSense device           //
#define FORMAT RS2_FORMAT_RGB8  // rs2_format identifies how binary data is encoded within a frame      //
#define WIDTH 640               // Defines the number of columns for each frame                         //
#define HEIGHT 480              // Defines the number of lines for each frame                           //
#define FPS 60                  // Defines the rate of frames per second                                //
#define STREAM_INDEX 0          // Defines the stream index, used for multiple streams of the same type //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int64_t lastFrameID = 0;

int main(int argc, char **argv)
{
  rs2_error *e = 0;

  // Create a context object. This object owns the handles to all connected realsense devices.
  // The returned object should be released with rs2_delete_context(...)
  rs2_context *ctx = rs2_create_context(RS2_API_VERSION, &e);
  check_error(e);

  /* Get a list of all the connected devices. */
  // The returned object should be released with rs2_delete_device_list(...)
  rs2_device_list *device_list = rs2_query_devices(ctx, &e);
  check_error(e);

  int dev_count = rs2_get_device_count(device_list, &e);
  check_error(e);
  printf("There are %d connected RealSense devices.\n", dev_count);
  if (0 == dev_count)
    return EXIT_FAILURE;

  // Get the first connected device
  // The returned object should be released with rs2_delete_device(...)
  rs2_device *dev = rs2_create_device(device_list, 0, &e);
  check_error(e);

  // print_device_info(dev);

  // Create a pipeline to configure, start and stop camera streaming
  // The returned object should be released with rs2_delete_pipeline(...)
  rs2_pipeline *pipeline = rs2_create_pipeline(ctx, &e);
  check_error(e);

  // Create a config instance, used to specify hardware configuration
  // The retunred object should be released with rs2_delete_config(...)
  rs2_config *config = rs2_create_config(&e);
  check_error(e);

  // Request a specific configuration
  rs2_config_enable_stream(config, STREAM, STREAM_INDEX, WIDTH, HEIGHT, FORMAT, FPS, &e);
  check_error(e);

  // Start the pipeline streaming
  // The retunred object should be released with rs2_delete_pipeline_profile(...)
  rs2_pipeline_profile *pipeline_profile = rs2_pipeline_start_with_config(pipeline, config, &e);
  if (e)
  {
    printf("The connected device doesn't support color streaming!\n");
    exit(EXIT_FAILURE);
  }
  OpenConnection();
  while (!IsConnected)
    millisleep(100);

  printf("Connected.");
  LEAP_DEVICE_INFO *deviceProps = GetDeviceProperties();
  if (deviceProps)
    printf("Device Serial: %s.\n", deviceProps->serial);

  FILE *fp;
  fp = fopen("hand_tracking_data.txt", "w");
  //fprintf(fp, "frame_id hand_id hand_type palm_position_x palm_position_y palm_position_z\n");
  long long int start_time = LeapGetNow();
  for (;;)
  {
    // This call waits until a new composite_frame is available
    // composite_frame holds a set of frames. It is used to prevent frame drops
    // The returned object should be released with rs2_release_frame(...)
    rs2_frame *frames = rs2_pipeline_wait_for_frames(pipeline, RS2_DEFAULT_TIMEOUT, &e);
    check_error(e);

    // Returns the number of frames embedded within the composite frame
    int num_of_frames = rs2_embedded_frames_count(frames, &e);
    check_error(e);

    int i;
    for (i = 0; i < num_of_frames; ++i)
    {
      // The retunred object should be released with rs2_release_frame(...)
      rs2_frame *frame = rs2_extract_frame(frames, i, &e);
      check_error(e);

      const uint8_t *rgb_frame_data = (const uint8_t *)(rs2_get_frame_data(frame, &e));
      check_error(e);

      unsigned long long frame_number = rs2_get_frame_number(frame, &e);
      check_error(e);

      rs2_time_t frame_timestamp = rs2_get_frame_timestamp(frame, &e);
      check_error(e);

      // Specifies the clock in relation to which the frame timestamp was measured
      rs2_timestamp_domain frame_timestamp_domain = rs2_get_frame_timestamp_domain(frame, &e);
      check_error(e);
      const char *frame_timestamp_domain_str = rs2_timestamp_domain_to_string(frame_timestamp_domain);

      rs2_metadata_type frame_metadata_time_of_arrival = rs2_get_frame_metadata(frame, RS2_FRAME_METADATA_TIME_OF_ARRIVAL, &e);
      check_error(e);

      int i;
      LEAP_TRACKING_EVENT *leapframe = GetFrame();
      if (leapframe && (leapframe->tracking_frame_id > lastFrameID))
      {
        lastFrameID = leapframe->tracking_frame_id;

        if (!leapframe->nHands)
        {
          printf("%s", "No Hands detected!\n");
        }
        else
        {
          printf("% c\n", '.');

          // printf("Frame %lli with %i hands.\n", (long long int)leapframe->tracking_frame_id, leapframe->nHands);
          for (uint32_t h = 0; h < leapframe->nHands; h++)
          {
            LEAP_HAND *hand = &leapframe->pHands[h];
            fprintf(fp, "%lld %i %s %f %f %f %f %f %f %f \n",
                    frame_metadata_time_of_arrival,
                    hand->id,
                    (hand->type == eLeapHandType_Left ? "left" : "right"),
                    hand->palm.position.x,
                    hand->palm.position.y,
                    hand->palm.position.z,
                    hand->palm.orientation.w,
                    hand->palm.orientation.x,
                    hand->palm.orientation.y,
                    hand->palm.orientation.z);
          }
        }
      }
      char filename[300];
      sprintf(filename, "images/frame_%lld", frame_metadata_time_of_arrival);

      FILE *f = fopen(filename, "wb");
      fprintf(f, "P6\n%i %i 255\n", WIDTH, HEIGHT);
      int p = 0;
      for (int y = 0; y < WIDTH; y++)
      {
        for (int x = 0; x < HEIGHT; x += 1)
        {
          fputc(rgb_frame_data[p], f);     // 0 .. 255
          fputc(rgb_frame_data[p + 1], f); // 0 .. 255
          fputc(rgb_frame_data[p + 2], f); // 0 .. 255
          p += 3;
        }
      }
      fclose(f);
      rs2_release_frame(frame);
    }

    rs2_release_frame(frames);
  } // ctrl-c to exit
  fclose(fp);
  return 0;
}
