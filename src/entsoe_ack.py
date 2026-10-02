"""
What the platform says when it has nothing to say.

Asked for a period it holds no data for, ENTSO-E does not return an error. It
returns HTTP 200 and an `Acknowledgement_MarketDocument` — a perfectly ordinary,
protocol-defined reply meaning "the request was fine, there is simply nothing
here". Without recognising it, the parser gets a document whose `<type>` is
missing and reports `Expected documentType A85, got None`, which reads like a bug
in OUR code. It cost us one wrong diagnosis already: we concluded the request had
never reached the network, when in fact it had been answered.

That distinction is the whole point. "The source has no data" and "we are broken"
must never look alike in a log — working out which of the two you are looking at
is what eats the afternoons.

Lives in its own module because both parsers need it and neither should have to
import the other, and because it must stay free of `ingest` — that module raises
on import when there is no API key, and the parsers have to work offline.
"""

from lxml import etree

ACK_ROOT = "Acknowledgement_MarketDocument"


class NoDataAvailable(Exception):
    """The request was valid and the platform answered: there is nothing here.

    Not a failure of ours, and not a transport failure — callers that treat every
    exception as "the fetch broke" will mislabel it.
    """

    def __init__(self, doc_label: str, code: str | None, text: str | None):
        self.doc_label = doc_label
        self.code = code
        self.text = text
        detail = " / ".join(x for x in (code, text) if x) or "no reason given"
        super().__init__(f"{doc_label}: platform returned no data ({detail})")


def raise_if_acknowledgement(root: etree._Element, doc_label: str) -> None:
    """Turn an acknowledgement into NoDataAvailable; leave anything else alone."""
    if etree.QName(root).localname != ACK_ROOT:
        return
    reason = root.find("{*}Reason")
    code = reason.findtext("{*}code") if reason is not None else None
    text = reason.findtext("{*}text") if reason is not None else None
    raise NoDataAvailable(doc_label, code, text)
