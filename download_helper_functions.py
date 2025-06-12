# Donwloads helper functions from other sources
# Need to take 2 lists one with file names and another one with URLs of repos

def download_helper_functions(files: list, repos: list):
  for i, fd in enumerate(files):
    rl = repos[i]
    if Path(fd).is_file():
      print(f"{fd} already exists, moving on")
    else:
      print(f"Downloading: {fd}")
      try:
        request = requests.get(rl)
        request.raise_for_status()
        with open(fd, "wb") as f:
          f.write(request.content)
        print(f"Successfully downloaded {fd}")
      except requests.exceptions.RequestException as e:
        print(f"Error downloading {fd}: {e}")
