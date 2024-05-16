t = {
    1: {
        1: (200, 300),
        2: (100, 100),
        3: (10, 10),
        4: (1, 1)
    },
    2: {
        1: (3, 12),
        2: (121, 1),
        3: (100, 100),
        4: (23, 98)
    },
    3: {
        1: (32, 64),
        2: (13, 62),
        3: (74, 12),
        4: (53, 29)
    }
}

d = {
    1: {},
    2: {},
    3: {}
}

# for KP_id in d:
#     d[KP_id]
#     min_id = min(d[KP_id], key=lambda k: d[KP_id][k])
#     print("KP: ", KP_id, " min val id: ", min_id)

for KP_id in t:
    t[KP_id]
    min_id = min(t[KP_id], key=lambda k: t[KP_id][k])
    print("KP: ", KP_id, " min val id: ", min_id)
