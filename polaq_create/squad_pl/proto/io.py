"""Code taken from original SQuAD project source code."""

from google.protobuf.internal.decoder import _DecodeVarint32
from google.protobuf.internal.encoder import _EncodeVarint


from squad_pl.proto.dataset_pb2 import Article


def write_article(article: Article, fileobj):
    """
    Writes the given Article protocol buffer to the given file-like object.
    """
    msg = article.SerializeToString()
    # Write number of message's bytes as a header
    _EncodeVarint(fileobj.write, len(msg))
    fileobj.write(msg)


def read_article(fileobj):
    """
    Reads a single Article protocol buffer from the given file-like object.
    """
    hdr = fileobj.read(4)
    if len(hdr) == 0:
        return None
    msg_length, hdr_length = _DecodeVarint32(hdr, 0)
    msg = hdr[hdr_length:] + fileobj.read(msg_length - (4 - hdr_length))

    article = Article()
    article.ParseFromString(msg)
    return article


def read_articles(filename):
    """
    Reads all articles as a generator from the file with the given name.
    """
    articles = []
    with open(filename, "rb") as f:
        while True:
            article = read_article(f)
            if article is None:
                return articles
            yield article
