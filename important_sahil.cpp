
ll modadd(ll a,ll b){					//mod arithmetic
	a%=mod;
	b%=mod;
	ll ans=(a+b)%mod;
	return ans;
}

ll modmul(ll a,ll b){
	a%=mod;
	b%=mod;
	ll ans=(a*b)%mod;
	return ans;
}

ll modsub(ll a,ll b){
	a%=mod;
	b%=mod;
	ll ans=a-b;
	if(ans<0)
		ans+=mod;
	return ans;
}

					//Segment tree [Range based Summation]
vector<int> t(400020),a(100005);
void build(int v,int tl,int tr){
	if(tl==tr)
		t[v]=a[tl];
	else{
		int tm=(tl+tr)/2;
		build(2*v,tl,tm);
		build(2*v+1,tm+1,tr);
		t[v]=t[2*v]+t[2*v+1];
	}
}

int sum(int v,int tl,int tr,int l,int r){
	if(l>r)
		return 0;
	if(tl==l && tr==r)
		return t[v];
	int tm=(tl+tr)/2;
	return sum(2*v,tl,tm,l,min(r,tm))+sum(2*v+1,tm+1,tr,max(l,tm+1),r);
}

void update(int v,int tl,int tr,int pos,int val){
	if(tl==pos && tr==pos)
		t[v]=val;
	else{
		int tm=(tl+tr)/2;
		if(pos<=tm)
			update(2*v,tl,tm,pos,val);
		else
			update(2*v+1,tm+1,tr,pos,val);
		t[v]=t[2*v]+t[2*v+1];
	}
}


									//Lazy Seg tree [Range based update & query][sum]
vector<int> t(400020),lazy(400020,0),a(100005);
void build(int v,int tl,int tr){
	if(tl==tr)
		t[v]=a[tl];
	else{
		int tm=(tl+tr)/2;
		build(2*v,tl,tm);
		build(2*v+1,tm+1,tr);
		t[v]=0;
	}
}

void push(int v,int tl,int tr){
	if(lazy[v]){
		t[v] += lazy[v]*(tr-tl+1);
		if(tl!=tr){
			lazy[2*v] = lazy[v];
			lazy[2*v+1] = lazy[v];
		}
		lazy[v] = 0;
	}
}

int query(int v,int tl,int tr,int l,int r){
	push(v,tl,tr);
	if(l>r)
		return 0;
	if(tl==l && r==tr)
		return t[v];
	int tm=(tl+tr)/2;
	return query(2*v,tl,tm,l,min(r,tm))+query(2*v+1,tm+1,tr,max(l,tm+1),r);
}

void update(int v,int tl,int tr,int l,int r,int val){
	push(v,tl,tr);
	if(l>r)
		return ;
	if(tl==l && tr==r){
		t[v]+=(tr-tl+1)*val;
		if(tl!=tr){
			lazy[2*v] = val;
			lazy[2*v+1] = val;
		}
	}else{
		int tm=(tl+tr)/2;
		update(2*v,tl,tm,l,min(r,tm),val);
		update(2*v+1,tm+1,tr,max(l,tm+1),r,val);
		t[v]=t[2*v]+t[2*v+1];
	}
}


vector<vector<pii>> adj;				//Dijktras
void dijktras(int s,vector<int> &d,vector<int> &p){
	int n=adj.size();
	d.assign(n,INT_MAX);
	p.assign(n,-1);
	priority_queue< pii,vector<pii>,greater<pii> > q;
	d[s]=0;
	q.push({0,s});

	while(!q.empty()){
		int v=q.top().sec;
		int d_v=q.top().fi;
		q.pop();
		if(d_v!=d[v])
			continue;
		
		for(auto node:adj[v]){
			int to=node.fi;
			int len=node.sec;
			if(d[v]+len<d[to]){
				d[to]=d[v]+len;
				p[to]=v;
				q.push({d[to],to});
			}
		}
	}
	
}

vector<vector<ll>> adj;				//DFS
vector<bool> mark;
void dfs(ll v){
	mark[v]=true;
	for(auto node:adj[v]){
		if(!mark[node]){
			dfs(node);
		}
	}
}


vector<ll> parent,size;							//DSU
void make_set(ll u){
	parent[u]=u;
	size[u]=1;
}
ll find_set(ll v){
	if(v==parent[v])
		return v;
	parent[v]=find_set(parent[v]);
	return parent[v];
}
void union_set(ll a,ll b){
	a=find_set(a);
	b=find_set(b);
	if(a!=b){
		if(size[b]>size[a])
			swap(a,b);
		parent[b]=a;
		size[a]+=size[b];
	}
}
for(int i=1;i<=n;i++)
	make_set(i);




typedef struct uvw{						//Kruskal
	ll u,v,w;
}uvw;

bool f(uvw a,uvw b){
	return a.w<b.w;
}

void kruskal(){
	ll n,edges;
	in n>>edges;
	for(int i=1;i<=n;i++)
		make_set(i);
	vector<uvw> v;
	while(edges--){
		uvw temp;
		in temp.u>>temp.v>>temp.w;
		v.pb(temp);
	}
	sort(all(v),f);
	ll cost=0;
	vector<uvw> result;
	for(auto edge:v){
		if(find_set(edge.u)!=find_set(edge.v)){
			union_build(edge.u,edge.v);
			result.pb(edge);
			cost+=edge.w;
		}
	}
	for(auto edge:result){
		out edge.u<<" "<<edge.v<<"\t"<<edge.w<<"\n";
	}
	w(cost);
}


ll n,timer,l;								//lca - binary lifting
vector<vector<ll>> adj,up;
vector<ll> tin,tout;

void dfs(ll v,ll p){
	tin[v]=++timer;
	up[v][0]=p;
	for(int i=1;i<=l;i++)
		up[v][i]=up[up[v][i-1]][i-1];
	for(auto u:adj[v])
		if(u!=p)
			dfs(u,v);
	tout[v]=++timer;
}

void pre(ll root){
	l=ceil(log2(n));
	timer=0;
	up.clear();
	up.assign(n+1,vector<ll>(l+1));
	tin.resize(n+1);
	tout.resize(n+1);
	dfs(root,root);
}

bool is_ancestor(ll u,ll v){
	return tin[u]<=tin[v] && tout[u]>=tout[v];
}

ll lca(ll u,ll v){
	if(is_ancestor(u,v))
		return u;
	if(is_ancestor(v,u))
		return v;
	for(int i=l;i>=0;i--)
		if(!is_ancestor(up[u][i],v))
			u=up[u][i];
	return up[u][0];
}


vector<vector<ll>> adj;								//Offline bridges
vector<ll> tin,low;
vector<bool> visited;
ll n,timer;

void dfs(ll v,ll p){
	visited[v]=true;
	tin[v]=low[v]=++timer;
	for(ll u:adj[v]){
		if(u==p)
			continue;
		if(visited[u]){
			low[v]=min(low[v],tin[u]);
		}else{
			dfs(u,v);
			if(tin[v]<low[u])
				out u<<" "<<v<<"\n";
			low[v]=min(low[v],low[u]);
		}
	}
}
void pre(){
	timer=0;
	tin.resize(n+1);
	low.resize(n+1);
	visited.assign(n+1,false);
	for(int i=1;i<=n;i++)
		if(!visited[i])
			dfs(i,i);
}




vvll adj,trans;							// strongly connected components...Tarjon's algorithm
vbool visited;
vll stacks,component;
void dfs1(ll v){
	visited[v]=true;
	for(auto u:adj[v])
		if(!visited[u])
			dfs1(u);
	stacks.pb(v);
}
void dfs2(ll v){
	visited[v]=true;
	for(auto u:trans[v])
		if(!visited[u])
			dfs2(u);
	component.pb(v);
}

int main(){
	ll n,m;
	in n>>m;
	adj.resize(n+1);
	trans.resize(n+1);
	while(m--){
		ll u,v;
		in u>>v;
		adj[u].pb(v);
		trans[v].pb(u);
	}
	visited.assign(n+1,false);
	for(int i=1;i<=n;i++)
		if(!visited[i])
			dfs1(i);
	visited.assign(n+1,false);

	for(int i=n-1;i>=0;i--)
		if(!visited[stacks[i]]){
			component.clear();
			dfs2(stacks[i]);
			PRINT component;
		}
	return 0;
}

vbool prime(1e7+5,true);							//Sieve (prime or not)
void se(){
	int n=1e7+2;
	prime[0]=prime[1]=false;
	for(int i=2;i*i<n;i++)
		if(prime[i])
			for(int j=i*i;j<n;j+=i)
				prime[j]=false;
}


void floyd(){		//d[i][i]=0 , if there is a edge with weight W b/w i & j, then d[i][j]=W else d[i][j]=inmax
	for(int k=0;k<n;k++){
		for(int i=0;i<n;i++){
			for(int j=0;j<n;j++){
				if(d[i][j]>d[i][k]+d[k][j]){
					d[i][j]=d[i][k]+d[k][j];
				}
			}
		}
	}
}








