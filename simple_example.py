from image_to_stl import ImageTo3D

converter = ImageTo3D(max_height=15.0, base_thickness=3.0)

image_path = "sample_bg_removed.png"
output_path = "sample_output.stl"

depth_map, mask = converter.process_image_to_stl(
    image_path=image_path,
    output_path=output_path,
    method='gradient',
    resolution_factor=0.3,
    remove_bg=True
)