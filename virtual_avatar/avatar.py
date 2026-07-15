from furhat_remote_api import FurhatRemoteAPI

# Statuses furhat.listen() returns instead of recognized speech
_NO_SPEECH_STATUSES = {"SILENCE", "INTERRUPTED", "FAILED"}


def furhat_connect(host="127.0.0.1") -> FurhatRemoteAPI:
    """Build a client for the Furhat robot at `host`."""
    return FurhatRemoteAPI(host)


class FurhatAudioCapture:
    def __init__(self, furhat: FurhatRemoteAPI):
        self.furhat = furhat

    def capture_response(self) -> str:
        status = self.furhat.listen()
        message = getattr(status, "message", "") or ""
        return "" if message in _NO_SPEECH_STATUSES else message
