# Person 2 - 3D Dental Model Preparation

# TUSHAR THAKUR

## Dataset

Source:
Teeth3DS - 3D Teeth Scan Segmentation and Labeling Challenge

For the current demo, only Dataset Part 1 has been processed.

## Current Processing Status

| Dataset | Upper | Lower |
|---|---:|---:|
| Part 1 | 150 cases | 150 cases |

All 300 cases were processed successfully.

## Output Structure

processed_part1_upper/
    CASE_ID/
        tooth_11.obj
        tooth_12.obj
        ...
        metadata.json

processed_part1_lower/
    CASE_ID/
        tooth_31.obj
        tooth_32.obj
        ...
        metadata.json

Not every case contains every tooth.

## Tooth Naming

The tooth filenames use FDI numbering.

Examples:

Upper:
- tooth_11.obj
- tooth_12.obj
- tooth_21.obj
- tooth_26.obj

Lower:
- tooth_31.obj
- tooth_32.obj
- tooth_41.obj
- tooth_46.obj

## metadata.json

Each case contains metadata such as:

- patient ID
- jaw
- original vertex count
- original triangle count
- tooth FDI number
- Teeth3DS instance ID
- processed vertex count
- processed triangle count
- output OBJ filename

## Important Coordinate Information

The extracted tooth meshes retain the original coordinates from the Teeth3DS OBJ.

The teeth are NOT centered individually.

Therefore, their positions relative to the original dental arch are preserved.

This is important for 2D–3D registration.

## Important

The current demo uses only Part 1.

Parts 2–6 will be processed later.

The original Teeth3DS dataset files are NOT included in this repository.

Only the processed demonstration outputs are included.

## Person 3 Input

Person 3 can directly load the tooth OBJ files using:

- Open3D
- Trimesh
- PyVista
- Blender
- Three.js loaders

Example:

processed_part1_upper/BUGB9DY1/tooth_11.obj

The complete set of individual teeth can be loaded together to reconstruct the dental arch.