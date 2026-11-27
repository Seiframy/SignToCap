"""Optional Windows Credential Manager storage for the scoped RunPod key."""
import ctypes as ct
from ctypes import wintypes as wt

TARGET = "SignToCap/RunPod-Pods"
GENERIC = 1
LOCAL_MACHINE = 2  # Windows persists the credential for this user on this machine.
NOT_FOUND = 1168


class Credential(ct.Structure):
    _fields_ = [("Flags", wt.DWORD), ("Type", wt.DWORD), ("TargetName", wt.LPWSTR),
                ("Comment", wt.LPWSTR), ("LastWritten", wt.FILETIME), ("BlobSize", wt.DWORD),
                ("Blob", ct.POINTER(ct.c_ubyte)), ("Persist", wt.DWORD), ("AttributeCount", wt.DWORD),
                ("Attributes", ct.c_void_p), ("TargetAlias", wt.LPWSTR), ("UserName", wt.LPWSTR)]


def windows_api():
    api = ct.WinDLL("Advapi32", use_last_error=True)
    api.CredReadW.argtypes = (wt.LPCWSTR, wt.DWORD, wt.DWORD, ct.POINTER(ct.POINTER(Credential)))
    api.CredWriteW.argtypes = (ct.POINTER(Credential), wt.DWORD)
    api.CredDeleteW.argtypes = (wt.LPCWSTR, wt.DWORD, wt.DWORD)
    api.CredFree.argtypes = (ct.c_void_p,)
    api.CredFree.restype = None
    return api


def load_key():
    api = windows_api()
    item = ct.POINTER(Credential)()
    if not api.CredReadW(TARGET, GENERIC, 0, ct.byref(item)):
        if ct.get_last_error() == NOT_FOUND:
            return None
        raise OSError("Windows Credential Manager could not read the RunPod key")
    try:
        return ct.string_at(item.contents.Blob, item.contents.BlobSize).decode("ascii")
    finally:
        api.CredFree(item)


def save_key(key):
    api = windows_api()
    size = len(key)
    blob = (ct.c_ubyte * size).from_buffer_copy(key.encode("ascii"))
    item = Credential(Type=GENERIC, TargetName=TARGET, BlobSize=size, Blob=blob,
                      Persist=LOCAL_MACHINE, UserName="SignToCap")
    try:
        if not api.CredWriteW(ct.byref(item), 0):
            raise OSError("Windows Credential Manager could not save the RunPod key")
    finally:
        ct.memset(blob, 0, size)


def forget_key():
    api = windows_api()
    if not api.CredDeleteW(TARGET, GENERIC, 0) and ct.get_last_error() != NOT_FOUND:
        raise OSError("Windows Credential Manager could not remove the RunPod key")
