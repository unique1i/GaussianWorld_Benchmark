import os
import re

# Directory containing log files
log_dir = '/home/yli7/projects/gaussian_world/preprocess_2d_language_feature/logs'

# Output file for failed scene names
output_file = '/home/yli7/projects/gaussian_world/preprocess_2d_language_feature/splits/matterport3d_failed.txt'

# Regex to capture scene name from the 'Processing scene' line
scene_pattern = re.compile(r'Processing scene\s+(?P<scene>\S+)')

failed_scenes = set()

# Walk through all files in the log directory
for root, dirs, files in os.walk(log_dir):
    for fname in files:
        fpath = os.path.join(root, fname)
        try:
            with open(fpath, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Could not read file {fpath}: {e}")
            continue

        # Search for 'Traceback (most recent call last)' occurrences
        for i, line in enumerate(lines):
            if 'Traceback (most recent call last)' in line:
                # Look two lines above for the scene info
                for offset in (1, 2):
                    idx = i - offset
                    if idx >= 0:
                        match = scene_pattern.search(lines[idx])
                        if match:
                            failed_scenes.add(match.group('scene'))
                            break  # stop after first match for this traceback

# Write unique failed scenes to the output file
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, 'w') as out_f:
    for scene in sorted(failed_scenes):
        out_f.write(scene + '\n')

print(f"Extracted {len(failed_scenes)} failed scenes to: {output_file}")
