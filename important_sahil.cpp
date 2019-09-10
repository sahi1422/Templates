
ll modadd(ll a,ll b){		//mod arithmetic
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






