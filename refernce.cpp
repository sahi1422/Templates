//all binary strings of length n
ll poww(ll n){
    ll ans=1;
    for(ll i=0;i<n;i++)
        ans*=2;
    return ans;
}
vector<string>arr;
void foo(ll len){
    ll size=poww(len);
    ll j,i;
    for(j=0;j<size;j++){
        for(i=0;i<len;i++){
            if(j&(1<<i))
                s+="1";
            else
                s+="0";
        }
        arr.pb(s);
    }
}

//occurance count
vector<ll>occ;
    for(i=0;i<n-1;i++){
        if(arr[i]==arr[i+1])
            count++;
        else{
            occ.pb(count);
            count=1;
        }
    }
occ.pb(count);

//fib logn
map<long, long> F;
F[0]=F[1]=1;
ll f(ll n) {
    if (F.count(n)) return F[n];
    ll k=n/2;
    if (n%2==0) { // n=2*k
        return F[n] = (f(k)*f(k) + f(k-1)*f(k-1)) % M;
    } else { // n=2*k+1
        return F[n] = (f(k)*f(k+1) + f(k-1)*f(k)) % M;
    }
}

//modpow
long long modpow(long long x, long long n,long long M) {
    int result=1;
    while(n>0) {
        if(n % 2 ==1)
            result=(result * x)%M;
        x=(x*x)%M;
        n=n/2;
    }
    return result;
}

//fastpower no mod
double fpow(double base, ll exp) {
    if (exp == 0)
        return 1;
    else if (exp == 1)
        return base;
    else if ((exp & 1) != 0)
        return base * fpow(base * base, exp / 2);
    else
        return fpow(base * base, exp / 2);
}

//modadd
ll modadd(ll a,ll b){
    ll ans=(a%mod+b%mod)%mod;
    if(ans<0)
        ans+=mod;
    return ans;
}

//invmod
ll invmod(ll a){
    return (modpow(a,mod-2,mod));
}

//prime generation
vector<ll> prime;
prime.pb(2);
for (i = 3; i < 200000; i += 2){
	for (j = 0; j < prime.size(); j++){
		if (i%prime[j] == 0) break;
			if (prime[j]*prime[j] > i){
				prime.pb(i);
				break;
			}
		}
	}

//prime factors
map<ll, ll> factors;
cin >> num;
for (j = 0; j < prime.size(); j++) {
	while (num % prime[j] == 0) {
		num /= prime[j];
		factors[prime[j]]++;
	}
	if (num == 1)
		break;
}
if (num > 1)
	factors[num]++;

//normal sieve
void sieve(){
    primes[0]=primes[1]=0;
    for(int i=2;i*i<=MAX;i++){
        for(int j=i*i;j<=MAX;j+=i)
            primes[j]=0;
    }
}

//log(n) factors walk sieve, stores min prime no
vector<bool>v(MAX+1,false);
vector<ll>sp(MAX+1);
void sieve1(){
    for (ll i = 2; i <= MAX; i += 2)
        sp[i] = 2;
    for (ll i = 3; i <= MAX; i += 2){
        if (!v[i]){
            sp[i] = i;
            for (ll j = i; (j*i) <= MAX; j += 2){
                if (!v[j*i]) {
                    v[j * i] = true;
                    sp[j * i] = i;
                }
            }
        }
    }
}

//prime fact using spf
        map<ll, ll> factors;
        num=x;
        if (num == 1)
            goto l1;
        while (num % sp[num] == 0) {
            factors[sp[num]]++;
            num /= sp[num];
            if (num == 1)
                break;
        }
        l1:;
//smallest prime factor of i
vector<ll>lprime(1000006,0);
lprime[1] = 1;
for(int i=2;i<1000006;i++)
	if(!lprime[i])
		for(int j=i;j<1000006;j+=i)
			if(!lprime[j])
				lprime[j] = i;

//ncr mod m O(r)
ll ncr(ll n,ll k) {
    if(n<k) return 0;
    if(n==k) return 1;
    if(k==1) return n;
    if(k==0) return 1;
    k = min(k,n-k);
    ll ans =1;
    for(ll i=1;i<k+1;i++)
    {
        ans = ((ans*n)%mod*invmod(i))%mod;
        n--;
    }
    return ans;
}

//ncr mod m, m prime
ll nCrModpDP(ll n, ll r, ll p) {
    ll C[r+1];
    memset(C, 0, sizeof(C));
    C[0] = 1;
    for (ll i = 1; i <= n; i++) {
        for (ll j = min(i, r); j > 0; j--)
            C[j] = (C[j] + C[j-1])%p;
    }
    return C[r];
}
ll ncr(ll n, ll r, ll p) {
    if (r==0)
        return 1;
    ll ni = n%p, ri = r%p;
    return (ncr(n/p, r/p, p) *nCrModpDP(ni, ri, p)) % p;
}

//summation
ll summation(ll n, ll pow){
    ll ans;
    n=n%mod;
    if(pow==3){
        ans=((((n*n)%mod*(n+1)%mod)%mod*(n+1)%mod)%mod*invmod(4))%mod;
    }
    else if(pow==2){
        ans=(((n*(n+1)%mod)%mod*((2*n)%mod+1)%mod)%mod*invmod(6))%mod;
    }
    else if(pow==1){
        ans=((n*(n+1)%mod)%mod*invmod(2))%mod;
    }
    return ans;
}

//segtree
vector<int>tree(400020),arr(100005);
int n,k;
void build(int node, int start, int end){//1,1,n
    if(start == end)
        tree[node] = A[start];
    else{
        int mid = (start + end) / 2;
        build(2*node, start, mid);
        build(2*node+1, mid+1, end);
        tree[node] = tree[2*node] + tree[2*node+1];
    }
}
void update(int node, int start, int end, int idx, int val){//1,1,n,i,val          i is 1 based
    if(start == end){
        arr[idx] += val;
        tree[node] += val;
    }
    else{
        int mid = (start + end) / 2;
        if(start <= idx && idx <= mid)
            update(2*node, start, mid, idx, val);
        else
            update(2*node+1, mid+1, end, idx, val);
        tree[node] = tree[2*node] + tree[2*node+1];
    }
}
int query(int node, int start, int end, int l, int r){//1,1,n,l,r            l,r is 1 based
    if(r < start || end < l)
        return 0;
    if(l <= start && end <= r)
        return tree[node];
    int mid = (start + end) / 2;
    int p1 = query(2*node, start, mid, l, r);
    int p2 = query(2*node+1, mid+1, end, l, r);
    return (p1 + p2);
}

//BIT 
vector<ll>BIT(10000005,0),arr(200005,0);
ll n,mx;
void initialize(){
	for(int i=0;i<200005;i++)
		arr[i] = 0;
	for(int i=0;i<10000005;i++)
		BIT[i] = 0;
}
ll query(ll ind){
	ll ans = 0;
	for(;ind>0;ind -= (ind&-ind)){
		ans += BIT[ind];
	}
	return ans;
}
ll update(ll ind){
	for(;ind<=mx;ind += (ind&-ind)){
		BIT[ind]++;
	}
}

//bfs, shortest path 0,1 edges
vector< vector<ll> >graph(300000);
vector<ll>dist(300000,LLONG_MAX);
void bfs(ll src) {
    deque <ll> Q;
    dist[src] = 0;
    Q.pb(src);
    while (!Q.empty()) {
        ll v = Q.front();
        Q.pop_front();
        for (auto e:graph[v]) {
            if (dist[e] > dist[v] + e.weight) {
                dist[e] = dist[v] + e.weight;
                if (e.weight == 0)
                    Q.push_front(e);
                else
                    Q.push_back(e);
            }
        }
    }
}

//cyclic check
bool isCyclicUtil(ll v, bool visited1[], bool *recStack) {
    if(visited1[v] == false) {
        visited1[v] = true;
        recStack[v] = true;
        for(auto e:graph[v]) {
            if ( !visited1[e] && isCyclicUtil(e, visited1, recStack) )
                return true;
            else if (recStack[e])
                return true;
        }
    }
    recStack[v] = false;
    return false;
}
bool isCyclic(){
    bool *visited1 = new bool[500001];
    bool *recStack = new bool[500001];
    for(ll i = 0; i <= 500000; i++) {
        visited1[i] = false;
        recStack[i] = false;
    }
    for(ll i = 1; i <= n; i++)
        if (isCyclicUtil(i, visited1, recStack))
            return true;
    return false;
}

//Dijikstra Algorithm
void shortestPath(int src,vector<int>&dist,int flag){
    priority_queue< pair<int,int>, vector <pair<int,int> > , greater<pair<int,int> > > pq;
    fill(dist.begin(),dist.end(),INT_MAX);
    pq.push(make_pair(0, src));
    dist[src] = 0;
    while (!pq.empty()){
        int u = pq.top().second;
        int ve = pq.top().first;
        pq.pop();
        if(ve>dist[u])
        	continue;
        for (auto e:adj[u]){
        	int v = e.x;
        	int weight = e.y;
        	if (dist[v] > dist[u] + weight){
        		dist[v] = dist[u] + weight;
        		pq.push(make_pair(dist[v], v));
        	}
        }
    }
}

//Topological Sorting
vector<int>adj[MAXN],visited(MAXN,0),indeg(MAXN,0),ans;
void topological(){
	for(int i=0;i<MAXN;i++)
		for(int j=0;j<adj[i].size();j++)
			indeg[v[i][j]]=0;
	queue<int>q;
	for(int i=0;i<MAXN;i++)
		if(indeg[i]==0){
			q.push(i);
			visited[i]=true;
		}
	while(!q.empty()){
		int p=q.front();
		q.pop();
		ans.pb(p);
		for(int i=0;i<adj[p].size();i++)
			if(!visited[adj[p][i]]){
				indeg[adj[p][i]]--;
				if(!indeg[adj[p][i]]){
					q.push(adj[p][i]);
					visited[adj[p][i]]=true;
				}
			}
	}
}

//sqrt-decomposition
void precal(){
	int ind = -1;
	ma = LLONG_MIN;
	sz = sqrt(n+.0)+1;
	int j=0,k=0;
	for(int i=0;i<n;i++){
		if(j==sz){
			j = 0;
			k++;
		}
		b[k]+=a[i];
		j++;
	}
}
ll query(ll p,ll q,ll x){
	p--;
	q--;
	ll ans = 0;
	ll c_l = p/sz;
	ll c_r = q/sz;
	if(c_l == c_r){
		for(int i=p;i<=q;i++)
			ans+=a[i];
	}
	else{
		if(p%sz!=0)
			c_l++;
		for(int i = p;i<c_l*sz;i++)
			ans+=a[i];
		ll i = c_l*sz;
		while(i+sz-1<=q){
			ll r = i/sz;
			ans+=b[r];
			i+=sz;
		}
		while(i<=q){
			ans+=a[i];
			i++;
		}
	}
	return ans;
}
void update(ll i,ll x,ll y){
	i--;
	a[i] = x;
	ll num = i/sz;
	b[num]+=(x-y);
}

//SCC
vector<ll>adj[400005],adjr[400005];
vector<ll>visited(400005,0),visitedr(400005,0);
vector<ll>order,component;
void dfs1(ll src){
	visited[src] = 1;
	for(auto e:adj[src])
		if(!visited[e])
			dfs1(e);
	order.pb(src);
}
void dfs2(ll src){
	visitedr[src] = 1;
	component.pb(src);
	for(auto e:adjr[src])
		if(!visitedr[e])
			dfs2(e);
}
int main(){
	ll n,m;
	cin >> n >> m; // n - no. of vertices , m - no. of edges
	for(int i=0;i<n;i++){
		ll a, b;
		cin >> a >> b;
		a--; // converting to 0-based
		b--; // converting to 0-based
		adj[a].push_back(b);
		adjr[b].push_back(a);
	}
	for(int i=0;i<n;++i)
		if(!visited[i])
			dfs1(i);
	for(int i=0;i<n;++i){
		ll v = order[n-1-i]; 
		if(!visitedr[v]){
			dfs2 (v);
			component.clear();
		}
	}
}

// MO's Algorithm (Query sqrt-decomposition)
ll block; //sqrt(N)
struct QUERY{
	ll L,R;
};
bool compare(QUERY a,QUERY b){
	if(a.L/block != b.L/block)
		return (a.L/block) < (b.L/block);
	return a.R<b.R;
}
void mo(vector<ll>a,vector<QUERY>q){
	block = sqrt(a.size());
	sort(q.begin(),q.end(),compare);
	ll curL=0,curR=0,curSum=0;
	for(int i=0;i<q.size();i++){
		ll L = q[i].L,R = q[i].R;
		while(curL<L){
			curSum-=a[curL];
			curL++;
		}
		while(curL>L){
			curSum+=a[curL];
			curL--;
		}
		while(curR<=R){
			curSum+=a[curR];
			curR++;
		}
		while(curR>(R+1)){
			curSum-=a[curR-1];
			curR--;
		}
		cout << curSum << "\n"; 
	}
}

//Lazy segment tree
vector<ll>tree(400020),lazy(400020,0),arr(100005);
ll n,k;
void build(ll node, ll start, ll end) { //1,1,n
    if(start == end)
        tree[node] = arr[start];
    else {
        ll mid = (start + end) / 2;
        build(2*node, start, mid);
        build(2*node + 1, mid + 1, end);
        tree[node] = min(tree[2*node], tree[2*node + 1]);
    }
}
void update(ll node, ll start, ll end, ll l, ll r, ll val) { //1,1,n,l,r,val		l,r is 1-based
    if(lazy[node]) {
        tree[node] += lazy[node];
        if(start != end) {
            lazy[2*node] += lazy[node];
            lazy[2*node + 1] += lazy[node];
        }
        lazy[node] = 0;
    }
    if(start > end || start > r || end < l)
        return;
    if(l <= start && end <= r) {
        tree[node] += val;
        if(start != end) {
            lazy[2*node] += val;
            lazy[2*node + 1] += val;
        }
        return;
    }
    ll mid = (start + end) / 2;
    update(2*node, start, mid, l, r, val);
    update(2*node + 1, mid + 1, end, l, r, val);
    tree[node] = min(tree[2*node], tree[2*node + 1]);
}
ll query(ll node, ll start, ll end, ll l, ll r) { // 1,1,n,l,r			l,r is 1-based
    if(start > end || start > r || end < l)
        return INT_MAX;
    if(lazy[node]) {
        tree[node] += lazy[node];
        if(start != end) {
            lazy[2*node] += lazy[node];
            lazy[2*node + 1] += lazy[node];
        }
        lazy[node] = 0;
    }
    if(l <= start && end <= r)
        return tree[node];
    ll mid = (start + end) / 2;
    ll p1 = query(2*node, start, mid, l, r);
    ll p2 = query(2*node + 1, mid + 1, end, l, r);
    return min(q1,q2);
}

//Matrix Exponentiation
vector< vector<ll> > matmul(vector< vector<ll> >A,vector< vector<ll> >B, ll m) // multiply matrix A,B
{
	vector< vector<ll> >ans(m,vector<ll>(m,0));
	for(int i=0;i<m;i++)
		for(int j=0;j<m;j++)
			for(int k=0;k<m;k++)
				ans[i][j] = modadd(ans[i][j],modmul(A[i][k],B[k][j]));
	return ans;
}
vector< vector<ll> > matpow(vector< vector<ll> >mat,ll n,ll m) // calculate mat^n with dimension of math is m*m
{
	if(n==1)
		return mat;
	if(n&1)
		return matmul(mat,matpow(mat,n-1,m),m);
	vector< vector<ll> >temp = matpow(mat,n/2,m);
	return matmul(temp,temp,m);
}

//Z-algorithm
vector<ll>z;
void zfunc(string s){ // calculates z value at index i such that maximum prefix length for string p starting from index i
  ll sz = s.size();
  z.pb(-1);
  ll l=0,r=0;
  for(int i=1;i<sz;i++){
    if(i>r){
      l=i;
      r=i;
      while(r<sz && s[r-l]==s[r])
        r++;
      z.pb(r-l);
      r--;
    }
    else{
      ll k = i-l;
      if(z[k]<r-i+1)
        z.pb(z[k]);
      else{
        l=i;
        while(r<sz && s[r-l]==s[r])
          r++;
        z.pb(r-l);
        r--;
      }
    }
  }
}

//LCA - binary lifting
vector< pll >adj[100005];
vector<ll>tin,tout;
vector< vector<ll> >up;
ll timer,l,ans,n;
void initialize(ll n){
    tin.clear();
    tout.clear();
    tin.resize(n);
    tout.resize(n);
    timer = 0;
    l = ceil(log2(n));
    up.assign(n,vector<ll>(l+1));
}
void dfs(ll src,ll par){
    tin[src] = ++timer;
    up[src][0] = par;
    for(int i=1;i<=l;i++)
        up[src][i] = up[up[src][i-1]][i-1];
    for(auto e:adj[src])
        if(e!=par)
            dfs(e,src);
    tout[src] = ++timer;
}
bool is_ancestor(ll u,ll v){
    return (tin[u]<=tin[v] && tout[u]>=tout[v]);
}
ll lca(ll u,ll v){
    if(is_ancestor(u,v))
        return u;
    if(is_ancestor(v,u))
        return v;
    for(int i=l;i>=0;i--)
        if(!is_ancestor(up[u][i],v))
            u = up[u][i];
    return up[u][0];
}

//Bridges - offline
vector< vector<ll> >adj;
vector<ll>tin,low,visited;
ll timer;
void dfs(ll src,ll p = -1){
	visited[src]++;
	tin[src] = ++timer;
	low[src] = timer;
	for(auto e:adj[src])
		if(e==p)
			continue;
		else if(visited[e])
			low[src] = min(low[src],tin[e]);
		else{
			dfs(e,src);
			low[src] = min(low[src],low[e]);
			if(low[e]>tin[src])
				cout << src << " " << e << "\n";
		}
}
void bridge(ll n){
	tin.assign(n+1,-1);
	low.assign(n+1,-1);
	visited.assign(n+1,0);
	timer = 0;
	for(int i=1;i<=n;i++)
		if(!visited[i])
			dfs(i);
}

//Articulation Points
vector< vector<ll> >adj;
vector<ll>tin,low,visited;
ll timer;
void dfs(ll src,ll p = -1)
{
	ll child = 0;
	visited[src]++;
	tin[src] = ++timer;
	low[src] = timer;
	for(auto e:adj[src])
		if(e==p);
		else if(visited[e])
			low[src] = min(low[src],tin[e]);
		else
		{
			dfs(e,src);
			low[src] = min(low[src],low[e]);
			if(low[e]>=tin[src] && p!=-1)
				cout << src << "\n";
			child++;
		}
	if(p==-1 && child>1)
		cout << src << "\n";
}
void bridge(ll n)
{
	tin.assign(n+1,-1);
	low.assign(n+1,-1);
	visited.assign(n+1,0);
	timer = 0;
	for(int i=1;i<=n;i++)
		if(!visited[i])
			dfs(i);
}

// angle of point (xc,yc) with respect to (0,0)
ld RadToDeg(ld rad)
{
	ll deg = rad*180.0/3.1415926535;
	return deg;
}
ld angle = 90.0;
if(xc!=0)
	angle = RadToDeg(atan(abs(yc)/abs(xc)));
if(xc<0 && yc>0)
	angle = 180 - angle;
if(xc<0 && yc<=0)
	angle = 180 + angle;
if(xc>=0 && yc<0)
	angle = 360 - angle;
