function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%


% Hence, sum(sum(R.*M)) is the sum of all the elements of M for 
% which the corresponding element in R equals 1.

costRegTheta = lambda * sum(sum(Theta.^2)) / 2;
costRegX = lambda * sum(sum(X.^2)) / 2;
J = sum(sum(R.*((X*Theta' - Y).^2)))/2;
J += costRegTheta + costRegX;

#Compute X Grad
for MovieI=1:num_movies

  #Looking for each user rating movie i
  idxUser = find(R(MovieI, :)==1);
  
  ThetaTemp = Theta(idxUser,:);
  YTemp = Y(MovieI, idxUser);

  X_grad(MovieI,:) = (X(MovieI,:)*ThetaTemp'-YTemp) * ThetaTemp + (lambda*X(MovieI,:));
  #Theta_grad(idxUser, :) = (X(idxUser,:)*ThetaTemp'-YTemp) * X(idxUser,:);
endfor

#Compute Theta Grad
for UserI= 1:num_users
  #Looking for each movie whom user has been rated
  idxMovie = find(R(:, UserI)==1);
  ThetaTemp = Theta(UserI,:); # OK
  YTemp = Y(idxMovie, UserI)';
  oo = X(idxMovie, :)*ThetaTemp'-YTemp';
  xx = X(idxMovie, :);
  dd = oo'*xx;
  #YTemp' * X
  #YTemp = Y(UserI, idxMovie)
  Theta_grad(UserI,:) = dd + (lambda * ThetaTemp);
endfor




% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
