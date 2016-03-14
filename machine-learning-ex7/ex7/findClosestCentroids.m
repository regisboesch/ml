function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

for m=1:size(X,1)
  #iterate over centroids
  idx_centreoids = -1;
  val_centroids= 10000000;
  
  for centroid=1:K
    v = sum((X(m,:)-centroids(centroid,:)).**2);
    #take minimum
    if v < val_centroids
      val_centroids = v;
      idx_centreoids = centroid;
    endif;
  endfor;
  
  #asign minimum
  idx(m) = idx_centreoids;
endfor;






% =============================================================

end

