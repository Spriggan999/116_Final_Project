"""Stable scene identities and a training-first split across both catalogs."""

CATALOGS = ('mixed', 'procthor', 'hssd')


def scene_specs(catalog='mixed'):
    """(catalog, native index, stable ID); mixed holds out 8 ProcTHOR + 4 HSSD.

    ProcTHOR IDs remain 0..199; HSSD IDs are 200..324, including when
    collecting a single catalog. Offsets address positions in this list.
    """
    procthor = [('procthor', i, i) for i in range(200)]
    hssd = [('hssd', i, 200+i) for i in range(125)]
    if catalog == 'procthor':
        return procthor
    if catalog == 'hssd':
        return hssd
    if catalog == 'mixed':
        # Distribute both training catalogs throughout a rotation. A short run
        # must not spend its entire budget in ProcTHOR before seeing any HSSD.
        training = [(i/192, row) for i,row in enumerate(procthor[:192])]
        training += [(i/121, row) for i,row in enumerate(hssd[:121])]
        training.sort(key=lambda pair:pair[0])
        return [row for _,row in training] + procthor[192:] + hssd[121:]
    raise ValueError(f'Unknown scene catalog {catalog!r}; choose from {CATALOGS}')


def scene_references(catalog='mixed'):
    """Load only the selected native catalogs; no simulator imports at startup."""
    specs = scene_specs(catalog)
    references = {}
    if any(name == 'procthor' for name, _, _ in specs):
        from hyperlux.datasets.procthor import Train200
        references['procthor'] = Train200().references
    if any(name == 'hssd' for name, _, _ in specs):
        from hyperlux.datasets.hssd import Train125
        references['hssd'] = Train125().references
    return [(name, index, scene_id, references[name][index])
            for name, index, scene_id in specs]
