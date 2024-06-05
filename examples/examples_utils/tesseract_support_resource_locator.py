import os
import re
import traceback
from tesseract_robotics.tesseract_common import ResourceLocator, SimpleLocatedResource

class TesseractSupportResourceLocator(ResourceLocator):
    def __init__(self):
        super().__init__()
        # print("TesseractSupportResourceLocator init -----------")
    
    def locateResource(self, url):
        # print("locateResource called -------")
        try:
            try:
                if os.path.exists(url):
                    return SimpleLocatedResource(url, url, self)
            except:
                pass
            url_match = re.match(r"^package:\/\/tesseract_support\/(.*)$",url)
            if (url_match is None):
                print("url_match failed")
                return None
            if not "TESSERACT_RESOURCE_PATH" in os.environ:
                return None
            tesseract_support = os.environ["TESSERACT_RESOURCE_PATH"]
            filename = os.path.join(tesseract_support, os.path.normpath(url_match.group(1)))
            ret = SimpleLocatedResource(url, filename, self)
            return ret
        except:
            traceback.print_exc()