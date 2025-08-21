import os
import re
import pandas as pd
from glob import glob

def load_subtitles_dataset(dataset_path):
    """
    Load cleaned .srt subtitle files into a pandas DataFrame.

    Args:
        dataset_path (str): Path to the folder containing .srt subtitle files.

    Returns:
        pd.DataFrame: DataFrame with columns [season, episode, script].
    """
    subtitles_paths = glob(os.path.join(dataset_path, "*.srt"))

    scripts = []
    seasons = []
    episodes = []

    for path in subtitles_paths:
        filename = os.path.basename(path)

        # Extract season and episode from filename pattern like 1x01, 2x03, etc.
        match = re.search(r"(\d+)x(\d{2})", filename)
        if not match:
            continue

        season = int(match.group(1))
        episode = int(match.group(2))

        with open(path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            # remove empty lines and strip
            lines = [line.strip() for line in lines if line.strip()]
            script = " ".join(lines)

        scripts.append(script)
        seasons.append(season)
        episodes.append(episode)

    df = pd.DataFrame.from_dict({
        "season": seasons,
        "episode": episodes,
        "script": scripts
    })

    # Sort properly: first by season, then by episode
    df = df.sort_values(by=["season", "episode"]).reset_index(drop=True)

    return df
