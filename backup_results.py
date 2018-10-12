"""
A Dropbox-API helper to upload files and directories.
IMPORTANT: It always has to be initialized with `init()` before uploading.
"""

from helpers import config
import dropbox
from dropbox.files import WriteMode
from dropbox.exceptions import ApiError, AuthError
import sys
import os



def init():
    # Check for an access token
    TOKEN = config.get("DROPBOX_TOKEN")
    if (not TOKEN or len(TOKEN) == 0):
        sys.exit("ERROR: Looks like you didn't add your access token. "
            "Open up backup-and-restore-example.py in a text editor and "
            "paste in your token in line 14.")

    # Create an instance of a Dropbox class, which can make requests to the API.
    print("Creating a Dropbox object...")
    global dbx
    dbx = dropbox.Dropbox(TOKEN)

    # Check that the access token is valid
    try:
        dbx.users_get_current_account()
    except AuthError as err:
        sys.exit("ERROR: Invalid access token; try re-generating an access token from the app console on the web.")


def upload_file(src_path, dest_path):
    """Reference: https://stackoverflow.com/a/37399658/1381666"""
    # Max file-size where `dbx.files_upload` instead of an upload-session is used
    CHUNK_SIZE = 4 * 1024 * 1024

    with open(src_path, 'rb') as f:
        file_size = os.path.getsize(src_path)

        if file_size <= CHUNK_SIZE:
            print(dbx.files_upload(f.read(), dest_path, mode=WriteMode('add')))

        else:
            upload_session_start_result = dbx.files_upload_session_start(f.read(CHUNK_SIZE))
            cursor = dropbox.files.UploadSessionCursor(session_id=upload_session_start_result.session_id, offset=f.tell())
            commit = dropbox.files.CommitInfo(path=dest_path)

            while f.tell() < file_size:
                if ((file_size - f.tell()) <= CHUNK_SIZE):
                    print(dbx.files_upload_session_finish(f.read(CHUNK_SIZE), cursor, commit))
                else:
                    dbx.files_upload_session_append(f.read(CHUNK_SIZE), cursor.session_id, cursor.offset)
                    cursor.offset = f.tell()


def upload_dir(src_path, dest_path):
    """Reference: https://stackoverflow.com/a/29189818/1381666"""
    for root, _, files in os.walk(src_path):
        for filename in files:
            if filename.startswith('.'): continue

            local_path = os.path.join(root, filename)
            relative_path = os.path.relpath(local_path, src_path)
            dropbox_path = os.path.join(dest_path, relative_path)

            print(f"Uploading '{local_path}'..")
            upload_file(local_path, dropbox_path)



if __name__ == "__main__":
    init()
    upload_dir('./results', '/')
