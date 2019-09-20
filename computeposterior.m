function logprob = computeposterior(Xr, Xq, prior, mu, F, G, Sigma)

[bq, d, nq] = size(Xq);
[br, d, nr] = size(Xr);

logprob = zeros(nq, nr + 1);

% for each set of query frames, perform inference
for i=1:nq
    i
    xq = Xq(:, :, i);
    pobs = computeloglhoods(Xr, xq, prior, mu, F, G, Sigma);
    % compute denominator for posterior over models
    denom = logsumexp(pobs, 1);
    % compute Bayesian probabilities and store localization prob. for each
    % set of queries from new location
    bayesq = pobs - repmat(denom, length(pobs), 1);
    logprob(i, :) = exp(bayesq);
    exp(bayesq)
end

