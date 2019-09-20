function pobs = computeloglhoods(X, Xq, prior, mu, F, G, Sigma)
[b, d, n] = size(X);
% store prob. of obs given each model
pobs = zeros(n+1, 1);
% generate matrices for mvn params
[mup, A, Sp] = generateCompositeMatrices(X(:, :, 1), mu, F, G, Sigma);
Cov = A * A' + diag(Sp);
[mupq, Aq, Spq] = generateCompositeMatrices([X(:, :, 1); Xq], mu, F, G, Sigma);
Covq = Aq * Aq' + diag(Spq);
% generate covar params
[ld, R] = logdet(Cov, 'chol');
[ldq, Rq] = logdet(Covq, 'chol');
for i=1:n
    % compute loglikelihood of each location independently
    x = X(:, :, i);
    llh = logmvnpdf(reshape(x', [], 1), mup, ld, R);
    % compute loglikelihood of each location tied with query frames
    llhq = logmvnpdf(reshape([x; Xq]', [], 1), mupq, ldq, Rq);
    % compute log prob. of obs given model
    llhv = repmat(llh, n+1, 1);
    llhv(i+1) = llhq;
    pobs = pobs + llhv;
end
% calculate likelihood of query frames with indep. location
[mup, A, Sp] = generateCompositeMatrices(Xq, mu, F, G, Sigma);
Covqq = A * A' + diag(Sp);
% generate covar params
[ldqq, Rqq] = logdet(Covqq, 'chol');
llh0 = logmvnpdf(reshape(Xq', [], 1), mup, ldqq, Rqq);
pobs(1) = pobs(1) + llh0;
pobs = pobs + prior;
end