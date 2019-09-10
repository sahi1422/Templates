#include<bits/stdc++.h>
#define ll long long
#define pb push_back
#define all(a)                      a.begin(), a.end()
#define mod 1000000007
#define out cout<<
#define in cin>>
#define w(param) out "\n\t "<<#param<<" is : "<<param
#define fi first
#define sec second
#define pii pair<int,int>
#define maxn 100005
#define invv(from,to,v) for(int i=from;i<to;i++) \
						in v[i];
#define inv(v) for(int i=0;i<v.size();i++) \
				in v[i];
#define printv(v) out #v<<"is :\n"; \
			for(int i=0;i<v.size();i++) \
				out " "<<v[i];
using namespace std;
ll extgcd(ll a,ll b,ll& x,ll& y){if(b==0){x=1;y=0;return a;}else{int g=extgcd(b,a%b,y,x);y-=a/b*x;return g;}}
ll modpow(ll a,ll b){ll res=1;a%=mod;for(;b;b>>=1){if(b&1)res=res*a%mod;a=a*a%mod;}return res;}

int main() {
	ios_base::sync_with_stdio(false);
	cin.tie(NULL);	
	ll t;
	in t;
	while(t--){

	}
//	cout<<endl;
	return 0;
}
