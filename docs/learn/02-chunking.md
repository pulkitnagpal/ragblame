# 2. Chunking: cutting docs into pieces

Search works on pieces, not whole pages: a page about "Docker Desktop settings" covers fifty different
things, and only one paragraph of it answers a given question. So every RAG system first cuts its docs
into **chunks**.

The demo bot uses the most common default: every 220 words, cut, with 20 words of overlap so a fact
near a cut may still land whole in one of the two pieces. On our 287 Docker pages that makes
**1,219 chunks**.

That cutter doesn't know what a table is. **A real split from this repo**: the Docker Desktop settings
page has a table row about the "Docker subnet" setting. The 220-word cut lands in the middle of it:

```
chunk settings#10 ends:    | **Docker subnet** | Set a custom subnet to avoid conflicts with IPs in
                           your environment. Docker Desktop uses a private
chunk settings#11 starts:  ... uses a private IPv4 network for internal services, including a DNS
                           server and HTTP proxy. Defaul[t ...]
```

Neither piece holds the whole row. And the second piece starts mid-table with no header, so a reader
(or a model) can't tell which column "Default" belongs to. 10 table rows across the corpus get cut
like this.

**The fix most teams reach for: split on headings.** `heading_chunks` follows the page's own
structure, never cuts inside a table row, and starts each piece with its heading path. The same row now
sits whole in one chunk that begins:

```
Change your Docker Desktop settings > VM resources > Network
### Network
| Setting | Description | Platform |
```

That makes 2,367 smaller, self-describing chunks. Whether it actually fixes wrong answers is something
ragblame can measure (`ragblame fix heading-chunks`), rather than assume.
