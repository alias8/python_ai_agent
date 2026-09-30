"""Checks the knowledge base corpus and the eval's labels are well-formed —
a wrong label silently skews every eval number, so labels get tests too. No
embedding model is loaded here."""

from evals.kb_eval import load_queries

from agent_service.langchain.kb_corpus import load_articles


def test_articles_parse_with_unique_ids_and_real_bodies() -> None:
    articles = load_articles()
    assert len(articles) >= 20
    assert len({a.store_id for a in articles}) >= 2
    assert len({a.id for a in articles}) == len(articles)
    for article in articles:
        assert article.title
        assert len(article.body.split()) >= 250, article.id


def test_eval_queries_are_unique() -> None:
    # The same question may appear once per store.
    queries = [(q.query, q.store) for q in load_queries()]
    assert len(set(queries)) == len(queries)


def test_eval_labels_point_at_real_articles_and_real_evidence() -> None:
    articles = {a.id: a for a in load_articles()}
    for q in load_queries():
        assert q.category in ("paraphrase", "identifier", "unrelated", "other_store"), q.query
        if q.category in ("unrelated", "other_store"):
            assert not q.expected and q.evidence is None, q.query
            continue
        assert q.expected, q.query
        for article_id in q.expected:
            assert article_id in articles, (q.query, article_id)
        assert q.evidence is not None, q.query
        # The evidence quote must really be in the first expected article.
        assert q.evidence in articles[q.expected[0]].body, q.query
