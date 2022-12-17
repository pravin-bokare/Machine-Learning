import re
from pytube import Playlist
playlist = Playlist('https://youtube.com/playlist?list=PLeo1K3hjS3us_ELKYSj_Fth2tIEkdKXvV')
DOWNLOAD_DIR = 'E:\Data Science'
playlist._video_regex = re.compile(r"\"url\":\"(/watch\?v=[\w-]*)")
print(len(playlist.video_urls))
i=1
for url in playlist.video_urls:
    print(url)
for video in playlist.videos:
    print('downloading :{} {} with url : {}'.format(i, video.title, video.watch_url))
    video.streams.\
        filter(type='video', progressive=True, file_extension='mp4').\
        order_by('resolution').\
        desc().\
        first().\
        download(DOWNLOAD_DIR)
    i+=1