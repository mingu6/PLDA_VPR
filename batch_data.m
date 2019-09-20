function [train, summer_test, fall_test, winter_test, spring_test] = batch_data(ntest, ntrain, s, w)

o = ntest + floor(0.5 * ntest); % offset for train dataset

fall_train = fvecs_read('Nordland80k/densevlad4096_100/fall.fvecs', [o, o+ntrain-1])';
summer_train = fvecs_read('Nordland80k/densevlad4096_100/summer.fvecs', [o, o+ntrain-1])';
winter_train = fvecs_read('Nordland80k/densevlad4096_100/winter.fvecs', [o, o+ntrain-1])';
spring_train = fvecs_read('Nordland80k/densevlad4096_100/spring.fvecs', [o, o+ntrain-1])';

% load test data

fall_test = fvecs_read('Nordland80k/densevlad4096_100/fall.fvecs', ntest)';
summer_test = fvecs_read('Nordland80k/densevlad4096_100/summer.fvecs', ntest)';
winter_test = fvecs_read('Nordland80k/densevlad4096_100/winter.fvecs', ntest)';
spring_test = fvecs_read('Nordland80k/densevlad4096_100/spring.fvecs', ntest)';

% subsample and batch obs

idx = 1:s:ntrain - w;
idxs = [];
for i=1:w
    idxs = [idxs ; idx];
    idx = idx + 1;
end

[b, n] = size(idxs);
[nt, d] = size(fall_test);

% batched and subsampled training obs

fall_train1 = permute(reshape(fall_train(idxs, :), [b, n, d]), [1, 3, 2]);
summer_train1 = permute(reshape(summer_train(idxs, :), [b, n, d]), [1, 3, 2]);
winter_train1 = permute(reshape(winter_train(idxs, :), [b, n, d]), [1, 3, 2]);
spring_train1 = permute(reshape(spring_train(idxs, :), [b, n, d]), [1, 3, 2]);

train = cat(1, fall_train1, summer_train1, winter_train1, spring_train1);

% batched and subsampled test obs

idx = 1:s:ntest - w;
idxs = [];
for i=1:w
    idxs = [idxs ; idx];
    idx = idx + 1;
end

[b, n] = size(idxs);

fall_test = permute(reshape(fall_test(idxs, :), [b, n, d]), [1, 3, 2]);
summer_test = permute(reshape(summer_test(idxs, :), [b, n, d]), [1, 3, 2]);
winter_test = permute(reshape(winter_test(idxs, :), [b, n, d]), [1, 3, 2]);
spring_test = permute(reshape(spring_test(idxs, :), [b, n, d]), [1, 3, 2]);
end