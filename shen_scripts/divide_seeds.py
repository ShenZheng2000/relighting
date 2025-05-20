def format_seeds(input_str, count=None):
    # Parse and clean the input
    seeds = list(map(int, input_str.strip().splitlines()))
    
    # Just print all in one horizontal line if no group count
    if count is None:
        print(' '.join(map(str, seeds)))
        return
    
    # Split seeds into `count` nearly equal groups
    group_size = len(seeds) // count
    remainder = len(seeds) % count
    start = 0

    for i in range(count):
        end = start + group_size + (1 if i < remainder else 0)
        group = seeds[start:end]
        print(f"# Group {i}:")
        print(' '.join(map(str, group)))
        start = end


input_str = \
"""
3
6
10
14
17
19
21
22
26
27
31
32
34
39
40
48
51
53
55
58
59
61
62
63
74
75
76
77
78
79
80
83
84
85
86
87
88
97
98
100
107
108
111
112
115
116
117
118
119
121
122
124
127
128
129
130
132
134
138
141
146
148
152
159
161
162
164
168
173
175
176
188
193
197
200
203
206
208
213
215
216
219
224
228
232
235
237
244
249
250
261
263
267
273
275
277
280
281
282
283
284
286
287
290
291
296
300
301
304
310
312
314
325
326
327
328
336
339
343
345
351
352
353
354
355
356
359
362
363
364
373
376
377
380
384
395
398
400
402
404
405
407
408
411
413
414
415
417
419
424
425
426
427
436
438
441
445
448
450
452
454
455
459
460
462
466
467
468
470
472
474
476
478
481
482
483
486
488
490
492
493
497
498
"""


format_seeds(input_str, count=8) # count: numbers of groups (gpus)